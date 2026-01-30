import crypto from "node:crypto";
import fs from "node:fs";
import net from "node:net";
import os from "node:os";
import path from "node:path";
import { DEFAULT_AGENT_ID } from "../routing/session-key.js";
const DEFAULT_SECURITY = "deny";
const DEFAULT_ASK = "on-miss";
const DEFAULT_ASK_FALLBACK = "deny";
const DEFAULT_AUTO_ALLOW_SKILLS = false;
const DEFAULT_SOCKET = "~/.clawdbot/exec-approvals.sock";
const DEFAULT_FILE = "~/.clawdbot/exec-approvals.json";
export const DEFAULT_SAFE_BINS = ["jq", "grep", "cut", "sort", "uniq", "head", "tail", "tr", "wc"];
function hashExecApprovalsRaw(raw) {
    return crypto
        .createHash("sha256")
        .update(raw ?? "")
        .digest("hex");
}
function expandHome(value) {
    if (!value)
        return value;
    if (value === "~")
        return os.homedir();
    if (value.startsWith("~/"))
        return path.join(os.homedir(), value.slice(2));
    return value;
}
export function resolveExecApprovalsPath() {
    return expandHome(DEFAULT_FILE);
}
export function resolveExecApprovalsSocketPath() {
    return expandHome(DEFAULT_SOCKET);
}
function normalizeAllowlistPattern(value) {
    const trimmed = value?.trim() ?? "";
    return trimmed ? trimmed.toLowerCase() : null;
}
function mergeLegacyAgent(current, legacy) {
    const allowlist = [];
    const seen = new Set();
    const pushEntry = (entry) => {
        const key = normalizeAllowlistPattern(entry.pattern);
        if (!key || seen.has(key))
            return;
        seen.add(key);
        allowlist.push(entry);
    };
    for (const entry of current.allowlist ?? [])
        pushEntry(entry);
    for (const entry of legacy.allowlist ?? [])
        pushEntry(entry);
    return {
        security: current.security ?? legacy.security,
        ask: current.ask ?? legacy.ask,
        askFallback: current.askFallback ?? legacy.askFallback,
        autoAllowSkills: current.autoAllowSkills ?? legacy.autoAllowSkills,
        allowlist: allowlist.length > 0 ? allowlist : undefined,
    };
}
function ensureDir(filePath) {
    const dir = path.dirname(filePath);
    fs.mkdirSync(dir, { recursive: true });
}
function ensureAllowlistIds(allowlist) {
    if (!Array.isArray(allowlist) || allowlist.length === 0)
        return allowlist;
    let changed = false;
    const next = allowlist.map((entry) => {
        if (entry.id)
            return entry;
        changed = true;
        return { ...entry, id: crypto.randomUUID() };
    });
    return changed ? next : allowlist;
}
export function normalizeExecApprovals(file) {
    const socketPath = file.socket?.path?.trim();
    const token = file.socket?.token?.trim();
    const agents = { ...file.agents };
    const legacyDefault = agents.default;
    if (legacyDefault) {
        const main = agents[DEFAULT_AGENT_ID];
        agents[DEFAULT_AGENT_ID] = main ? mergeLegacyAgent(main, legacyDefault) : legacyDefault;
        delete agents.default;
    }
    for (const [key, agent] of Object.entries(agents)) {
        const allowlist = ensureAllowlistIds(agent.allowlist);
        if (allowlist !== agent.allowlist) {
            agents[key] = { ...agent, allowlist };
        }
    }
    const normalized = {
        version: 1,
        socket: {
            path: socketPath && socketPath.length > 0 ? socketPath : undefined,
            token: token && token.length > 0 ? token : undefined,
        },
        defaults: {
            security: file.defaults?.security,
            ask: file.defaults?.ask,
            askFallback: file.defaults?.askFallback,
            autoAllowSkills: file.defaults?.autoAllowSkills,
        },
        agents,
    };
    return normalized;
}
function generateToken() {
    return crypto.randomBytes(24).toString("base64url");
}
export function readExecApprovalsSnapshot() {
    const filePath = resolveExecApprovalsPath();
    if (!fs.existsSync(filePath)) {
        const file = normalizeExecApprovals({ version: 1, agents: {} });
        return {
            path: filePath,
            exists: false,
            raw: null,
            file,
            hash: hashExecApprovalsRaw(null),
        };
    }
    const raw = fs.readFileSync(filePath, "utf8");
    let parsed = null;
    try {
        parsed = JSON.parse(raw);
    }
    catch {
        parsed = null;
    }
    const file = parsed?.version === 1
        ? normalizeExecApprovals(parsed)
        : normalizeExecApprovals({ version: 1, agents: {} });
    return {
        path: filePath,
        exists: true,
        raw,
        file,
        hash: hashExecApprovalsRaw(raw),
    };
}
export function loadExecApprovals() {
    const filePath = resolveExecApprovalsPath();
    try {
        if (!fs.existsSync(filePath)) {
            return normalizeExecApprovals({ version: 1, agents: {} });
        }
        const raw = fs.readFileSync(filePath, "utf8");
        const parsed = JSON.parse(raw);
        if (parsed?.version !== 1) {
            return normalizeExecApprovals({ version: 1, agents: {} });
        }
        return normalizeExecApprovals(parsed);
    }
    catch {
        return normalizeExecApprovals({ version: 1, agents: {} });
    }
}
export function saveExecApprovals(file) {
    const filePath = resolveExecApprovalsPath();
    ensureDir(filePath);
    fs.writeFileSync(filePath, `${JSON.stringify(file, null, 2)}\n`, { mode: 0o600 });
    try {
        fs.chmodSync(filePath, 0o600);
    }
    catch {
        // best-effort on platforms without chmod
    }
}
export function ensureExecApprovals() {
    const loaded = loadExecApprovals();
    const next = normalizeExecApprovals(loaded);
    const socketPath = next.socket?.path?.trim();
    const token = next.socket?.token?.trim();
    const updated = {
        ...next,
        socket: {
            path: socketPath && socketPath.length > 0 ? socketPath : resolveExecApprovalsSocketPath(),
            token: token && token.length > 0 ? token : generateToken(),
        },
    };
    saveExecApprovals(updated);
    return updated;
}
function normalizeSecurity(value, fallback) {
    if (value === "allowlist" || value === "full" || value === "deny")
        return value;
    return fallback;
}
function normalizeAsk(value, fallback) {
    if (value === "always" || value === "off" || value === "on-miss")
        return value;
    return fallback;
}
export function resolveExecApprovals(agentId, overrides) {
    const file = ensureExecApprovals();
    return resolveExecApprovalsFromFile({
        file,
        agentId,
        overrides,
        path: resolveExecApprovalsPath(),
        socketPath: expandHome(file.socket?.path ?? resolveExecApprovalsSocketPath()),
        token: file.socket?.token ?? "",
    });
}
export function resolveExecApprovalsFromFile(params) {
    const file = normalizeExecApprovals(params.file);
    const defaults = file.defaults ?? {};
    const agentKey = params.agentId ?? DEFAULT_AGENT_ID;
    const agent = file.agents?.[agentKey] ?? {};
    const wildcard = file.agents?.["*"] ?? {};
    const fallbackSecurity = params.overrides?.security ?? DEFAULT_SECURITY;
    const fallbackAsk = params.overrides?.ask ?? DEFAULT_ASK;
    const fallbackAskFallback = params.overrides?.askFallback ?? DEFAULT_ASK_FALLBACK;
    const fallbackAutoAllowSkills = params.overrides?.autoAllowSkills ?? DEFAULT_AUTO_ALLOW_SKILLS;
    const resolvedDefaults = {
        security: normalizeSecurity(defaults.security, fallbackSecurity),
        ask: normalizeAsk(defaults.ask, fallbackAsk),
        askFallback: normalizeSecurity(defaults.askFallback ?? fallbackAskFallback, fallbackAskFallback),
        autoAllowSkills: Boolean(defaults.autoAllowSkills ?? fallbackAutoAllowSkills),
    };
    const resolvedAgent = {
        security: normalizeSecurity(agent.security ?? wildcard.security ?? resolvedDefaults.security, resolvedDefaults.security),
        ask: normalizeAsk(agent.ask ?? wildcard.ask ?? resolvedDefaults.ask, resolvedDefaults.ask),
        askFallback: normalizeSecurity(agent.askFallback ?? wildcard.askFallback ?? resolvedDefaults.askFallback, resolvedDefaults.askFallback),
        autoAllowSkills: Boolean(agent.autoAllowSkills ?? wildcard.autoAllowSkills ?? resolvedDefaults.autoAllowSkills),
    };
    const allowlist = [
        ...(Array.isArray(wildcard.allowlist) ? wildcard.allowlist : []),
        ...(Array.isArray(agent.allowlist) ? agent.allowlist : []),
    ];
    return {
        path: params.path ?? resolveExecApprovalsPath(),
        socketPath: expandHome(params.socketPath ?? file.socket?.path ?? resolveExecApprovalsSocketPath()),
        token: params.token ?? file.socket?.token ?? "",
        defaults: resolvedDefaults,
        agent: resolvedAgent,
        allowlist,
        file,
    };
}
function isExecutableFile(filePath) {
    try {
        const stat = fs.statSync(filePath);
        if (!stat.isFile())
            return false;
        if (process.platform !== "win32") {
            fs.accessSync(filePath, fs.constants.X_OK);
        }
        return true;
    }
    catch {
        return false;
    }
}
function parseFirstToken(command) {
    const trimmed = command.trim();
    if (!trimmed)
        return null;
    const first = trimmed[0];
    if (first === '"' || first === "'") {
        const end = trimmed.indexOf(first, 1);
        if (end > 1)
            return trimmed.slice(1, end);
        return trimmed.slice(1);
    }
    const match = /^[^\s]+/.exec(trimmed);
    return match ? match[0] : null;
}
function resolveExecutablePath(rawExecutable, cwd, env) {
    const expanded = rawExecutable.startsWith("~") ? expandHome(rawExecutable) : rawExecutable;
    if (expanded.includes("/") || expanded.includes("\\")) {
        if (path.isAbsolute(expanded)) {
            return isExecutableFile(expanded) ? expanded : undefined;
        }
        const base = cwd && cwd.trim() ? cwd.trim() : process.cwd();
        const candidate = path.resolve(base, expanded);
        return isExecutableFile(candidate) ? candidate : undefined;
    }
    const envPath = env?.PATH ?? env?.Path ?? process.env.PATH ?? process.env.Path ?? "";
    const entries = envPath.split(path.delimiter).filter(Boolean);
    const hasExtension = process.platform === "win32" && path.extname(expanded).length > 0;
    const extensions = process.platform === "win32"
        ? hasExtension
            ? [""]
            : (env?.PATHEXT ??
                env?.Pathext ??
                process.env.PATHEXT ??
                process.env.Pathext ??
                ".EXE;.CMD;.BAT;.COM")
                .split(";")
                .map((ext) => ext.toLowerCase())
        : [""];
    for (const entry of entries) {
        for (const ext of extensions) {
            const candidate = path.join(entry, expanded + ext);
            if (isExecutableFile(candidate))
                return candidate;
        }
    }
    return undefined;
}
export function resolveCommandResolution(command, cwd, env) {
    const rawExecutable = parseFirstToken(command);
    if (!rawExecutable)
        return null;
    const resolvedPath = resolveExecutablePath(rawExecutable, cwd, env);
    const executableName = resolvedPath ? path.basename(resolvedPath) : rawExecutable;
    return { rawExecutable, resolvedPath, executableName };
}
export function resolveCommandResolutionFromArgv(argv, cwd, env) {
    const rawExecutable = argv[0]?.trim();
    if (!rawExecutable)
        return null;
    const resolvedPath = resolveExecutablePath(rawExecutable, cwd, env);
    const executableName = resolvedPath ? path.basename(resolvedPath) : rawExecutable;
    return { rawExecutable, resolvedPath, executableName };
}
function normalizeMatchTarget(value) {
    if (process.platform === "win32") {
        const stripped = value.replace(/^\\\\[?.]\\/, "");
        return stripped.replace(/\\/g, "/").toLowerCase();
    }
    return value.replace(/\\\\/g, "/").toLowerCase();
}
function tryRealpath(value) {
    try {
        return fs.realpathSync(value);
    }
    catch {
        return null;
    }
}
function globToRegExp(pattern) {
    let regex = "^";
    let i = 0;
    while (i < pattern.length) {
        const ch = pattern[i];
        if (ch === "*") {
            const next = pattern[i + 1];
            if (next === "*") {
                regex += ".*";
                i += 2;
                continue;
            }
            regex += "[^/]*";
            i += 1;
            continue;
        }
        if (ch === "?") {
            regex += ".";
            i += 1;
            continue;
        }
        regex += ch.replace(/[.*+?^${}()|[\\]\\\\]/g, "\\$&");
        i += 1;
    }
    regex += "$";
    return new RegExp(regex, "i");
}
function matchesPattern(pattern, target) {
    const trimmed = pattern.trim();
    if (!trimmed)
        return false;
    const expanded = trimmed.startsWith("~") ? expandHome(trimmed) : trimmed;
    const hasWildcard = /[*?]/.test(expanded);
    let normalizedPattern = expanded;
    let normalizedTarget = target;
    if (process.platform === "win32" && !hasWildcard) {
        normalizedPattern = tryRealpath(expanded) ?? expanded;
        normalizedTarget = tryRealpath(target) ?? target;
    }
    normalizedPattern = normalizeMatchTarget(normalizedPattern);
    normalizedTarget = normalizeMatchTarget(normalizedTarget);
    const regex = globToRegExp(normalizedPattern);
    return regex.test(normalizedTarget);
}
function resolveAllowlistCandidatePath(resolution, cwd) {
    if (!resolution)
        return undefined;
    if (resolution.resolvedPath)
        return resolution.resolvedPath;
    const raw = resolution.rawExecutable?.trim();
    if (!raw)
        return undefined;
    const expanded = raw.startsWith("~") ? expandHome(raw) : raw;
    if (!expanded.includes("/") && !expanded.includes("\\"))
        return undefined;
    if (path.isAbsolute(expanded))
        return expanded;
    const base = cwd && cwd.trim() ? cwd.trim() : process.cwd();
    return path.resolve(base, expanded);
}
export function matchAllowlist(entries, resolution) {
    if (!entries.length || !resolution?.resolvedPath)
        return null;
    const resolvedPath = resolution.resolvedPath;
    for (const entry of entries) {
        const pattern = entry.pattern?.trim();
        if (!pattern)
            continue;
        const hasPath = pattern.includes("/") || pattern.includes("\\") || pattern.includes("~");
        if (!hasPath)
            continue;
        if (matchesPattern(pattern, resolvedPath))
            return entry;
    }
    return null;
}
const DISALLOWED_PIPELINE_TOKENS = new Set([">", "<", "`", "\n", "\r", "(", ")"]);
/**
 * Iterates through a command string while respecting shell quoting rules.
 * The callback receives each character and the next character, and returns an action:
 * - "split": push current buffer as a segment and start a new one
 * - "skip": skip this character (and optionally the next via skip count)
 * - "include": add this character to the buffer
 * - { reject: reason }: abort with an error
 */
function iterateQuoteAware(command, onChar) {
    const parts = [];
    let buf = "";
    let inSingle = false;
    let inDouble = false;
    let escaped = false;
    let hasSplit = false;
    const pushPart = () => {
        const trimmed = buf.trim();
        if (trimmed) {
            parts.push(trimmed);
        }
        buf = "";
    };
    for (let i = 0; i < command.length; i += 1) {
        const ch = command[i];
        const next = command[i + 1];
        if (escaped) {
            buf += ch;
            escaped = false;
            continue;
        }
        if (!inSingle && !inDouble && ch === "\\") {
            escaped = true;
            buf += ch;
            continue;
        }
        if (inSingle) {
            if (ch === "'")
                inSingle = false;
            buf += ch;
            continue;
        }
        if (inDouble) {
            if (ch === '"')
                inDouble = false;
            buf += ch;
            continue;
        }
        if (ch === "'") {
            inSingle = true;
            buf += ch;
            continue;
        }
        if (ch === '"') {
            inDouble = true;
            buf += ch;
            continue;
        }
        const action = onChar(ch, next, i);
        if (typeof action === "object" && "reject" in action) {
            return { ok: false, reason: action.reject };
        }
        if (action === "split") {
            pushPart();
            hasSplit = true;
            continue;
        }
        if (action === "skip") {
            continue;
        }
        buf += ch;
    }
    if (escaped || inSingle || inDouble) {
        return { ok: false, reason: "unterminated shell quote/escape" };
    }
    pushPart();
    return { ok: true, parts, hasSplit };
}
function splitShellPipeline(command) {
    let emptySegment = false;
    const result = iterateQuoteAware(command, (ch, next) => {
        if (ch === "|" && next === "|") {
            return { reject: "unsupported shell token: ||" };
        }
        if (ch === "|" && next === "&") {
            return { reject: "unsupported shell token: |&" };
        }
        if (ch === "|") {
            emptySegment = true;
            return "split";
        }
        if (ch === "&" || ch === ";") {
            return { reject: `unsupported shell token: ${ch}` };
        }
        if (DISALLOWED_PIPELINE_TOKENS.has(ch)) {
            return { reject: `unsupported shell token: ${ch}` };
        }
        if (ch === "$" && next === "(") {
            return { reject: "unsupported shell token: $()" };
        }
        emptySegment = false;
        return "include";
    });
    if (!result.ok) {
        return { ok: false, reason: result.reason, segments: [] };
    }
    if (emptySegment || result.parts.length === 0) {
        return {
            ok: false,
            reason: result.parts.length === 0 ? "empty command" : "empty pipeline segment",
            segments: [],
        };
    }
    return { ok: true, segments: result.parts };
}
function tokenizeShellSegment(segment) {
    const tokens = [];
    let buf = "";
    let inSingle = false;
    let inDouble = false;
    let escaped = false;
    const pushToken = () => {
        if (buf.length > 0) {
            tokens.push(buf);
            buf = "";
        }
    };
    for (let i = 0; i < segment.length; i += 1) {
        const ch = segment[i];
        if (escaped) {
            buf += ch;
            escaped = false;
            continue;
        }
        if (!inSingle && !inDouble && ch === "\\") {
            escaped = true;
            continue;
        }
        if (inSingle) {
            if (ch === "'") {
                inSingle = false;
            }
            else {
                buf += ch;
            }
            continue;
        }
        if (inDouble) {
            if (ch === '"') {
                inDouble = false;
            }
            else {
                buf += ch;
            }
            continue;
        }
        if (ch === "'") {
            inSingle = true;
            continue;
        }
        if (ch === '"') {
            inDouble = true;
            continue;
        }
        if (/\s/.test(ch)) {
            pushToken();
            continue;
        }
        buf += ch;
    }
    if (escaped || inSingle || inDouble) {
        return null;
    }
    pushToken();
    return tokens;
}
function parseSegmentsFromParts(parts, cwd, env) {
    const segments = [];
    for (const raw of parts) {
        const argv = tokenizeShellSegment(raw);
        if (!argv || argv.length === 0) {
            return null;
        }
        segments.push({
            raw,
            argv,
            resolution: resolveCommandResolutionFromArgv(argv, cwd, env),
        });
    }
    return segments;
}
export function analyzeShellCommand(params) {
    // First try splitting by chain operators (&&, ||, ;)
    const chainParts = splitCommandChain(params.command);
    if (chainParts) {
        const chains = [];
        const allSegments = [];
        for (const part of chainParts) {
            const pipelineSplit = splitShellPipeline(part);
            if (!pipelineSplit.ok) {
                return { ok: false, reason: pipelineSplit.reason, segments: [] };
            }
            const segments = parseSegmentsFromParts(pipelineSplit.segments, params.cwd, params.env);
            if (!segments) {
                return { ok: false, reason: "unable to parse shell segment", segments: [] };
            }
            chains.push(segments);
            allSegments.push(...segments);
        }
        return { ok: true, segments: allSegments, chains };
    }
    // No chain operators, parse as simple pipeline
    const split = splitShellPipeline(params.command);
    if (!split.ok) {
        return { ok: false, reason: split.reason, segments: [] };
    }
    const segments = parseSegmentsFromParts(split.segments, params.cwd, params.env);
    if (!segments) {
        return { ok: false, reason: "unable to parse shell segment", segments: [] };
    }
    return { ok: true, segments };
}
export function analyzeArgvCommand(params) {
    const argv = params.argv.filter((entry) => entry.trim().length > 0);
    if (argv.length === 0) {
        return { ok: false, reason: "empty argv", segments: [] };
    }
    return {
        ok: true,
        segments: [
            {
                raw: argv.join(" "),
                argv,
                resolution: resolveCommandResolutionFromArgv(argv, params.cwd, params.env),
            },
        ],
    };
}
function isPathLikeToken(value) {
    const trimmed = value.trim();
    if (!trimmed)
        return false;
    if (trimmed === "-")
        return false;
    if (trimmed.startsWith("./") || trimmed.startsWith("../") || trimmed.startsWith("~"))
        return true;
    if (trimmed.startsWith("/"))
        return true;
    return /^[A-Za-z]:[\\/]/.test(trimmed);
}
function defaultFileExists(filePath) {
    try {
        return fs.existsSync(filePath);
    }
    catch {
        return false;
    }
}
export function normalizeSafeBins(entries) {
    if (!Array.isArray(entries))
        return new Set();
    const normalized = entries
        .map((entry) => entry.trim().toLowerCase())
        .filter((entry) => entry.length > 0);
    return new Set(normalized);
}
export function resolveSafeBins(entries) {
    if (entries === undefined)
        return normalizeSafeBins(DEFAULT_SAFE_BINS);
    return normalizeSafeBins(entries ?? []);
}
export function isSafeBinUsage(params) {
    if (params.safeBins.size === 0)
        return false;
    const resolution = params.resolution;
    const execName = resolution?.executableName?.toLowerCase();
    if (!execName)
        return false;
    const matchesSafeBin = params.safeBins.has(execName) ||
        (process.platform === "win32" && params.safeBins.has(path.parse(execName).name));
    if (!matchesSafeBin)
        return false;
    if (!resolution?.resolvedPath)
        return false;
    const cwd = params.cwd ?? process.cwd();
    const exists = params.fileExists ?? defaultFileExists;
    const argv = params.argv.slice(1);
    for (let i = 0; i < argv.length; i += 1) {
        const token = argv[i];
        if (!token)
            continue;
        if (token === "-")
            continue;
        if (token.startsWith("-")) {
            const eqIndex = token.indexOf("=");
            if (eqIndex > 0) {
                const value = token.slice(eqIndex + 1);
                if (value && (isPathLikeToken(value) || exists(path.resolve(cwd, value)))) {
                    return false;
                }
            }
            continue;
        }
        if (isPathLikeToken(token))
            return false;
        if (exists(path.resolve(cwd, token)))
            return false;
    }
    return true;
}
function evaluateSegments(segments, params) {
    const matches = [];
    const allowSkills = params.autoAllowSkills === true && (params.skillBins?.size ?? 0) > 0;
    const satisfied = segments.every((segment) => {
        const candidatePath = resolveAllowlistCandidatePath(segment.resolution, params.cwd);
        const candidateResolution = candidatePath && segment.resolution
            ? { ...segment.resolution, resolvedPath: candidatePath }
            : segment.resolution;
        const match = matchAllowlist(params.allowlist, candidateResolution);
        if (match)
            matches.push(match);
        const safe = isSafeBinUsage({
            argv: segment.argv,
            resolution: segment.resolution,
            safeBins: params.safeBins,
            cwd: params.cwd,
        });
        const skillAllow = allowSkills && segment.resolution?.executableName
            ? params.skillBins?.has(segment.resolution.executableName)
            : false;
        return Boolean(match || safe || skillAllow);
    });
    return { satisfied, matches };
}
export function evaluateExecAllowlist(params) {
    const allowlistMatches = [];
    if (!params.analysis.ok || params.analysis.segments.length === 0) {
        return { allowlistSatisfied: false, allowlistMatches };
    }
    // If the analysis contains chains, evaluate each chain part separately
    if (params.analysis.chains) {
        for (const chainSegments of params.analysis.chains) {
            const result = evaluateSegments(chainSegments, {
                allowlist: params.allowlist,
                safeBins: params.safeBins,
                cwd: params.cwd,
                skillBins: params.skillBins,
                autoAllowSkills: params.autoAllowSkills,
            });
            if (!result.satisfied) {
                return { allowlistSatisfied: false, allowlistMatches: [] };
            }
            allowlistMatches.push(...result.matches);
        }
        return { allowlistSatisfied: true, allowlistMatches };
    }
    // No chains, evaluate all segments together
    const result = evaluateSegments(params.analysis.segments, {
        allowlist: params.allowlist,
        safeBins: params.safeBins,
        cwd: params.cwd,
        skillBins: params.skillBins,
        autoAllowSkills: params.autoAllowSkills,
    });
    return { allowlistSatisfied: result.satisfied, allowlistMatches: result.matches };
}
/**
 * Splits a command string by chain operators (&&, ||, ;) while respecting quotes.
 * Returns null when no chain is present or when the chain is malformed.
 */
function splitCommandChain(command) {
    const parts = [];
    let buf = "";
    let inSingle = false;
    let inDouble = false;
    let escaped = false;
    let foundChain = false;
    let invalidChain = false;
    const pushPart = () => {
        const trimmed = buf.trim();
        if (trimmed) {
            parts.push(trimmed);
            buf = "";
            return true;
        }
        buf = "";
        return false;
    };
    for (let i = 0; i < command.length; i += 1) {
        const ch = command[i];
        if (escaped) {
            buf += ch;
            escaped = false;
            continue;
        }
        if (!inSingle && !inDouble && ch === "\\") {
            escaped = true;
            buf += ch;
            continue;
        }
        if (inSingle) {
            if (ch === "'")
                inSingle = false;
            buf += ch;
            continue;
        }
        if (inDouble) {
            if (ch === '"')
                inDouble = false;
            buf += ch;
            continue;
        }
        if (ch === "'") {
            inSingle = true;
            buf += ch;
            continue;
        }
        if (ch === '"') {
            inDouble = true;
            buf += ch;
            continue;
        }
        if (ch === "&" && command[i + 1] === "&") {
            if (!pushPart())
                invalidChain = true;
            i += 1;
            foundChain = true;
            continue;
        }
        if (ch === "|" && command[i + 1] === "|") {
            if (!pushPart())
                invalidChain = true;
            i += 1;
            foundChain = true;
            continue;
        }
        if (ch === ";") {
            if (!pushPart())
                invalidChain = true;
            foundChain = true;
            continue;
        }
        buf += ch;
    }
    const pushedFinal = pushPart();
    if (!foundChain)
        return null;
    if (invalidChain || !pushedFinal)
        return null;
    return parts.length > 0 ? parts : null;
}
/**
 * Evaluates allowlist for shell commands (including &&, ||, ;) and returns analysis metadata.
 */
export function evaluateShellAllowlist(params) {
    const chainParts = splitCommandChain(params.command);
    if (!chainParts) {
        const analysis = analyzeShellCommand({
            command: params.command,
            cwd: params.cwd,
            env: params.env,
        });
        if (!analysis.ok) {
            return {
                analysisOk: false,
                allowlistSatisfied: false,
                allowlistMatches: [],
                segments: [],
            };
        }
        const evaluation = evaluateExecAllowlist({
            analysis,
            allowlist: params.allowlist,
            safeBins: params.safeBins,
            cwd: params.cwd,
            skillBins: params.skillBins,
            autoAllowSkills: params.autoAllowSkills,
        });
        return {
            analysisOk: true,
            allowlistSatisfied: evaluation.allowlistSatisfied,
            allowlistMatches: evaluation.allowlistMatches,
            segments: analysis.segments,
        };
    }
    const allowlistMatches = [];
    const segments = [];
    for (const part of chainParts) {
        const analysis = analyzeShellCommand({
            command: part,
            cwd: params.cwd,
            env: params.env,
        });
        if (!analysis.ok) {
            return {
                analysisOk: false,
                allowlistSatisfied: false,
                allowlistMatches: [],
                segments: [],
            };
        }
        segments.push(...analysis.segments);
        const evaluation = evaluateExecAllowlist({
            analysis,
            allowlist: params.allowlist,
            safeBins: params.safeBins,
            cwd: params.cwd,
            skillBins: params.skillBins,
            autoAllowSkills: params.autoAllowSkills,
        });
        allowlistMatches.push(...evaluation.allowlistMatches);
        if (!evaluation.allowlistSatisfied) {
            return {
                analysisOk: true,
                allowlistSatisfied: false,
                allowlistMatches,
                segments,
            };
        }
    }
    return {
        analysisOk: true,
        allowlistSatisfied: true,
        allowlistMatches,
        segments,
    };
}
export function requiresExecApproval(params) {
    return (params.ask === "always" ||
        (params.ask === "on-miss" &&
            params.security === "allowlist" &&
            (!params.analysisOk || !params.allowlistSatisfied)));
}
export function recordAllowlistUse(approvals, agentId, entry, command, resolvedPath) {
    const target = agentId ?? DEFAULT_AGENT_ID;
    const agents = approvals.agents ?? {};
    const existing = agents[target] ?? {};
    const allowlist = Array.isArray(existing.allowlist) ? existing.allowlist : [];
    const nextAllowlist = allowlist.map((item) => item.pattern === entry.pattern
        ? {
            ...item,
            id: item.id ?? crypto.randomUUID(),
            lastUsedAt: Date.now(),
            lastUsedCommand: command,
            lastResolvedPath: resolvedPath,
        }
        : item);
    agents[target] = { ...existing, allowlist: nextAllowlist };
    approvals.agents = agents;
    saveExecApprovals(approvals);
}
export function addAllowlistEntry(approvals, agentId, pattern) {
    const target = agentId ?? DEFAULT_AGENT_ID;
    const agents = approvals.agents ?? {};
    const existing = agents[target] ?? {};
    const allowlist = Array.isArray(existing.allowlist) ? existing.allowlist : [];
    const trimmed = pattern.trim();
    if (!trimmed)
        return;
    if (allowlist.some((entry) => entry.pattern === trimmed))
        return;
    allowlist.push({ id: crypto.randomUUID(), pattern: trimmed, lastUsedAt: Date.now() });
    agents[target] = { ...existing, allowlist };
    approvals.agents = agents;
    saveExecApprovals(approvals);
}
export function minSecurity(a, b) {
    const order = { deny: 0, allowlist: 1, full: 2 };
    return order[a] <= order[b] ? a : b;
}
export function maxAsk(a, b) {
    const order = { off: 0, "on-miss": 1, always: 2 };
    return order[a] >= order[b] ? a : b;
}
export async function requestExecApprovalViaSocket(params) {
    const { socketPath, token, request } = params;
    if (!socketPath || !token)
        return null;
    const timeoutMs = params.timeoutMs ?? 15_000;
    return await new Promise((resolve) => {
        const client = new net.Socket();
        let settled = false;
        let buffer = "";
        const finish = (value) => {
            if (settled)
                return;
            settled = true;
            try {
                client.destroy();
            }
            catch {
                // ignore
            }
            resolve(value);
        };
        const timer = setTimeout(() => finish(null), timeoutMs);
        const payload = JSON.stringify({
            type: "request",
            token,
            id: crypto.randomUUID(),
            request,
        });
        client.on("error", () => finish(null));
        client.connect(socketPath, () => {
            client.write(`${payload}\n`);
        });
        client.on("data", (data) => {
            buffer += data.toString("utf8");
            let idx = buffer.indexOf("\n");
            while (idx !== -1) {
                const line = buffer.slice(0, idx).trim();
                buffer = buffer.slice(idx + 1);
                idx = buffer.indexOf("\n");
                if (!line)
                    continue;
                try {
                    const msg = JSON.parse(line);
                    if (msg?.type === "decision" && msg.decision) {
                        clearTimeout(timer);
                        finish(msg.decision);
                        return;
                    }
                }
                catch {
                    // ignore
                }
            }
        });
    });
}
