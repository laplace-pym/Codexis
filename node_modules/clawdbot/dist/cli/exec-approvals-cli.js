import fs from "node:fs/promises";
import JSON5 from "json5";
import { readExecApprovalsSnapshot, saveExecApprovals, } from "../infra/exec-approvals.js";
import { defaultRuntime } from "../runtime.js";
import { formatDocsLink } from "../terminal/links.js";
import { isRich, theme } from "../terminal/theme.js";
import { renderTable } from "../terminal/table.js";
import { callGatewayFromCli } from "./gateway-rpc.js";
import { describeUnknownError } from "./gateway-cli/shared.js";
import { nodesCallOpts, resolveNodeId } from "./nodes-cli/rpc.js";
function formatAge(msAgo) {
    const s = Math.max(0, Math.floor(msAgo / 1000));
    if (s < 60)
        return `${s}s`;
    const m = Math.floor(s / 60);
    if (m < 60)
        return `${m}m`;
    const h = Math.floor(m / 60);
    if (h < 24)
        return `${h}h`;
    const d = Math.floor(h / 24);
    return `${d}d`;
}
async function readStdin() {
    const chunks = [];
    for await (const chunk of process.stdin) {
        chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk)));
    }
    return Buffer.concat(chunks).toString("utf8");
}
async function resolveTargetNodeId(opts) {
    if (opts.gateway)
        return null;
    const raw = opts.node?.trim() ?? "";
    if (!raw)
        return null;
    return await resolveNodeId(opts, raw);
}
async function loadSnapshot(opts, nodeId) {
    const method = nodeId ? "exec.approvals.node.get" : "exec.approvals.get";
    const params = nodeId ? { nodeId } : {};
    const snapshot = (await callGatewayFromCli(method, opts, params));
    return snapshot;
}
function loadSnapshotLocal() {
    const snapshot = readExecApprovalsSnapshot();
    return {
        path: snapshot.path,
        exists: snapshot.exists,
        hash: snapshot.hash,
        file: snapshot.file,
    };
}
function saveSnapshotLocal(file) {
    saveExecApprovals(file);
    return loadSnapshotLocal();
}
async function loadSnapshotTarget(opts) {
    if (!opts.gateway && !opts.node) {
        return { snapshot: loadSnapshotLocal(), nodeId: null, source: "local" };
    }
    const nodeId = await resolveTargetNodeId(opts);
    const snapshot = await loadSnapshot(opts, nodeId);
    return { snapshot, nodeId, source: nodeId ? "node" : "gateway" };
}
function formatCliError(err) {
    const msg = describeUnknownError(err);
    return msg.includes("\n") ? msg.split("\n")[0] : msg;
}
function renderApprovalsSnapshot(snapshot, targetLabel) {
    const rich = isRich();
    const heading = (text) => (rich ? theme.heading(text) : text);
    const muted = (text) => (rich ? theme.muted(text) : text);
    const tableWidth = Math.max(60, (process.stdout.columns ?? 120) - 1);
    const file = snapshot.file ?? { version: 1 };
    const defaults = file.defaults ?? {};
    const defaultsParts = [
        defaults.security ? `security=${defaults.security}` : null,
        defaults.ask ? `ask=${defaults.ask}` : null,
        defaults.askFallback ? `askFallback=${defaults.askFallback}` : null,
        typeof defaults.autoAllowSkills === "boolean"
            ? `autoAllowSkills=${defaults.autoAllowSkills ? "on" : "off"}`
            : null,
    ].filter(Boolean);
    const agents = file.agents ?? {};
    const allowlistRows = [];
    const now = Date.now();
    for (const [agentId, agent] of Object.entries(agents)) {
        const allowlist = Array.isArray(agent.allowlist) ? agent.allowlist : [];
        for (const entry of allowlist) {
            const pattern = entry?.pattern?.trim() ?? "";
            if (!pattern)
                continue;
            const lastUsedAt = typeof entry.lastUsedAt === "number" ? entry.lastUsedAt : null;
            allowlistRows.push({
                Target: targetLabel,
                Agent: agentId,
                Pattern: pattern,
                LastUsed: lastUsedAt ? `${formatAge(Math.max(0, now - lastUsedAt))} ago` : muted("unknown"),
            });
        }
    }
    const summaryRows = [
        { Field: "Target", Value: targetLabel },
        { Field: "Path", Value: snapshot.path },
        { Field: "Exists", Value: snapshot.exists ? "yes" : "no" },
        { Field: "Hash", Value: snapshot.hash },
        { Field: "Version", Value: String(file.version ?? 1) },
        { Field: "Socket", Value: file.socket?.path ?? "default" },
        { Field: "Defaults", Value: defaultsParts.length > 0 ? defaultsParts.join(", ") : "none" },
        { Field: "Agents", Value: String(Object.keys(agents).length) },
        { Field: "Allowlist", Value: String(allowlistRows.length) },
    ];
    defaultRuntime.log(heading("Approvals"));
    defaultRuntime.log(renderTable({
        width: tableWidth,
        columns: [
            { key: "Field", header: "Field", minWidth: 8 },
            { key: "Value", header: "Value", minWidth: 24, flex: true },
        ],
        rows: summaryRows,
    }).trimEnd());
    if (allowlistRows.length === 0) {
        defaultRuntime.log("");
        defaultRuntime.log(muted("No allowlist entries."));
        return;
    }
    defaultRuntime.log("");
    defaultRuntime.log(heading("Allowlist"));
    defaultRuntime.log(renderTable({
        width: tableWidth,
        columns: [
            { key: "Target", header: "Target", minWidth: 10 },
            { key: "Agent", header: "Agent", minWidth: 8 },
            { key: "Pattern", header: "Pattern", minWidth: 20, flex: true },
            { key: "LastUsed", header: "Last Used", minWidth: 10 },
        ],
        rows: allowlistRows,
    }).trimEnd());
}
async function saveSnapshot(opts, nodeId, file, baseHash) {
    const method = nodeId ? "exec.approvals.node.set" : "exec.approvals.set";
    const params = nodeId ? { nodeId, file, baseHash } : { file, baseHash };
    const snapshot = (await callGatewayFromCli(method, opts, params));
    return snapshot;
}
function resolveAgentKey(value) {
    const trimmed = value?.trim() ?? "";
    return trimmed ? trimmed : "*";
}
function normalizeAllowlistEntry(entry) {
    const pattern = entry?.pattern?.trim() ?? "";
    return pattern ? pattern : null;
}
function ensureAgent(file, agentKey) {
    const agents = file.agents ?? {};
    const entry = agents[agentKey] ?? {};
    file.agents = agents;
    return entry;
}
function isEmptyAgent(agent) {
    const allowlist = Array.isArray(agent.allowlist) ? agent.allowlist : [];
    return (!agent.security &&
        !agent.ask &&
        !agent.askFallback &&
        agent.autoAllowSkills === undefined &&
        allowlist.length === 0);
}
export function registerExecApprovalsCli(program) {
    const formatExample = (cmd, desc) => `  ${theme.command(cmd)}\n    ${theme.muted(desc)}`;
    const approvals = program
        .command("approvals")
        .alias("exec-approvals")
        .description("Manage exec approvals (gateway or node host)")
        .addHelpText("after", () => `\n${theme.muted("Docs:")} ${formatDocsLink("/cli/approvals", "docs.clawd.bot/cli/approvals")}\n`);
    const getCmd = approvals
        .command("get")
        .description("Fetch exec approvals snapshot")
        .option("--node <node>", "Target node id/name/IP")
        .option("--gateway", "Force gateway approvals", false)
        .action(async (opts) => {
        try {
            const { snapshot, nodeId, source } = await loadSnapshotTarget(opts);
            if (opts.json) {
                defaultRuntime.log(JSON.stringify(snapshot));
                return;
            }
            const muted = (text) => (isRich() ? theme.muted(text) : text);
            if (source === "local") {
                defaultRuntime.log(muted("Showing local approvals."));
                defaultRuntime.log("");
            }
            const targetLabel = source === "local" ? "local" : nodeId ? `node:${nodeId}` : "gateway";
            renderApprovalsSnapshot(snapshot, targetLabel);
        }
        catch (err) {
            defaultRuntime.error(formatCliError(err));
            defaultRuntime.exit(1);
        }
    });
    nodesCallOpts(getCmd);
    const setCmd = approvals
        .command("set")
        .description("Replace exec approvals with a JSON file")
        .option("--node <node>", "Target node id/name/IP")
        .option("--gateway", "Force gateway approvals", false)
        .option("--file <path>", "Path to JSON file to upload")
        .option("--stdin", "Read JSON from stdin", false)
        .action(async (opts) => {
        try {
            if (!opts.file && !opts.stdin) {
                defaultRuntime.error("Provide --file or --stdin.");
                defaultRuntime.exit(1);
                return;
            }
            if (opts.file && opts.stdin) {
                defaultRuntime.error("Use either --file or --stdin (not both).");
                defaultRuntime.exit(1);
                return;
            }
            const { snapshot, nodeId, source } = await loadSnapshotTarget(opts);
            if (source === "local") {
                defaultRuntime.log(theme.muted("Writing local approvals."));
            }
            const targetLabel = source === "local" ? "local" : nodeId ? `node:${nodeId}` : "gateway";
            if (!snapshot.hash) {
                defaultRuntime.error("Exec approvals hash missing; reload and retry.");
                defaultRuntime.exit(1);
                return;
            }
            const raw = opts.stdin ? await readStdin() : await fs.readFile(String(opts.file), "utf8");
            let file;
            try {
                file = JSON5.parse(raw);
            }
            catch (err) {
                defaultRuntime.error(`Failed to parse approvals JSON: ${String(err)}`);
                defaultRuntime.exit(1);
                return;
            }
            file.version = 1;
            const next = source === "local"
                ? saveSnapshotLocal(file)
                : await saveSnapshot(opts, nodeId, file, snapshot.hash);
            if (opts.json) {
                defaultRuntime.log(JSON.stringify(next));
                return;
            }
            defaultRuntime.log(theme.muted(`Target: ${targetLabel}`));
            renderApprovalsSnapshot(next, targetLabel);
        }
        catch (err) {
            defaultRuntime.error(formatCliError(err));
            defaultRuntime.exit(1);
        }
    });
    nodesCallOpts(setCmd);
    const allowlist = approvals
        .command("allowlist")
        .description("Edit the per-agent allowlist")
        .addHelpText("after", () => `\n${theme.heading("Examples:")}\n${formatExample('clawdbot approvals allowlist add "~/Projects/**/bin/rg"', "Allowlist a local binary pattern for the main agent.")}\n${formatExample('clawdbot approvals allowlist add --agent main --node <id|name|ip> "/usr/bin/uptime"', "Allowlist on a specific node/agent.")}\n${formatExample('clawdbot approvals allowlist add --agent "*" "/usr/bin/uname"', "Allowlist for all agents (wildcard).")}\n${formatExample('clawdbot approvals allowlist remove "~/Projects/**/bin/rg"', "Remove an allowlist pattern.")}\n\n${theme.muted("Docs:")} ${formatDocsLink("/cli/approvals", "docs.clawd.bot/cli/approvals")}\n`);
    const allowlistAdd = allowlist
        .command("add <pattern>")
        .description("Add a glob pattern to an allowlist")
        .option("--node <node>", "Target node id/name/IP")
        .option("--gateway", "Force gateway approvals", false)
        .option("--agent <id>", 'Agent id (defaults to "*")')
        .action(async (pattern, opts) => {
        try {
            const trimmed = pattern.trim();
            if (!trimmed) {
                defaultRuntime.error("Pattern required.");
                defaultRuntime.exit(1);
                return;
            }
            const { snapshot, nodeId, source } = await loadSnapshotTarget(opts);
            if (source === "local") {
                defaultRuntime.log(theme.muted("Writing local approvals."));
            }
            const targetLabel = source === "local" ? "local" : nodeId ? `node:${nodeId}` : "gateway";
            if (!snapshot.hash) {
                defaultRuntime.error("Exec approvals hash missing; reload and retry.");
                defaultRuntime.exit(1);
                return;
            }
            const file = snapshot.file ?? { version: 1 };
            file.version = 1;
            const agentKey = resolveAgentKey(opts.agent);
            const agent = ensureAgent(file, agentKey);
            const allowlistEntries = Array.isArray(agent.allowlist) ? agent.allowlist : [];
            if (allowlistEntries.some((entry) => normalizeAllowlistEntry(entry) === trimmed)) {
                defaultRuntime.log("Already allowlisted.");
                return;
            }
            allowlistEntries.push({ pattern: trimmed, lastUsedAt: Date.now() });
            agent.allowlist = allowlistEntries;
            file.agents = { ...file.agents, [agentKey]: agent };
            const next = source === "local"
                ? saveSnapshotLocal(file)
                : await saveSnapshot(opts, nodeId, file, snapshot.hash);
            if (opts.json) {
                defaultRuntime.log(JSON.stringify(next));
                return;
            }
            defaultRuntime.log(theme.muted(`Target: ${targetLabel}`));
            renderApprovalsSnapshot(next, targetLabel);
        }
        catch (err) {
            defaultRuntime.error(formatCliError(err));
            defaultRuntime.exit(1);
        }
    });
    nodesCallOpts(allowlistAdd);
    const allowlistRemove = allowlist
        .command("remove <pattern>")
        .description("Remove a glob pattern from an allowlist")
        .option("--node <node>", "Target node id/name/IP")
        .option("--gateway", "Force gateway approvals", false)
        .option("--agent <id>", 'Agent id (defaults to "*")')
        .action(async (pattern, opts) => {
        try {
            const trimmed = pattern.trim();
            if (!trimmed) {
                defaultRuntime.error("Pattern required.");
                defaultRuntime.exit(1);
                return;
            }
            const { snapshot, nodeId, source } = await loadSnapshotTarget(opts);
            if (source === "local") {
                defaultRuntime.log(theme.muted("Writing local approvals."));
            }
            const targetLabel = source === "local" ? "local" : nodeId ? `node:${nodeId}` : "gateway";
            if (!snapshot.hash) {
                defaultRuntime.error("Exec approvals hash missing; reload and retry.");
                defaultRuntime.exit(1);
                return;
            }
            const file = snapshot.file ?? { version: 1 };
            file.version = 1;
            const agentKey = resolveAgentKey(opts.agent);
            const agent = ensureAgent(file, agentKey);
            const allowlistEntries = Array.isArray(agent.allowlist) ? agent.allowlist : [];
            const nextEntries = allowlistEntries.filter((entry) => normalizeAllowlistEntry(entry) !== trimmed);
            if (nextEntries.length === allowlistEntries.length) {
                defaultRuntime.log("Pattern not found.");
                return;
            }
            if (nextEntries.length === 0) {
                delete agent.allowlist;
            }
            else {
                agent.allowlist = nextEntries;
            }
            if (isEmptyAgent(agent)) {
                const agents = { ...file.agents };
                delete agents[agentKey];
                file.agents = Object.keys(agents).length > 0 ? agents : undefined;
            }
            else {
                file.agents = { ...file.agents, [agentKey]: agent };
            }
            const next = source === "local"
                ? saveSnapshotLocal(file)
                : await saveSnapshot(opts, nodeId, file, snapshot.hash);
            if (opts.json) {
                defaultRuntime.log(JSON.stringify(next));
                return;
            }
            defaultRuntime.log(theme.muted(`Target: ${targetLabel}`));
            renderApprovalsSnapshot(next, targetLabel);
        }
        catch (err) {
            defaultRuntime.error(formatCliError(err));
            defaultRuntime.exit(1);
        }
    });
    nodesCallOpts(allowlistRemove);
}
