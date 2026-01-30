import JSON5 from "json5";
import { readConfigFileSnapshot, writeConfigFile } from "../config/config.js";
import { danger, info } from "../globals.js";
import { defaultRuntime } from "../runtime.js";
import { formatDocsLink } from "../terminal/links.js";
import { formatCliCommand } from "./command-format.js";
import { theme } from "../terminal/theme.js";
import { shortenHomePath } from "../utils.js";
function isIndexSegment(raw) {
    return /^[0-9]+$/.test(raw);
}
function parsePath(raw) {
    const trimmed = raw.trim();
    if (!trimmed)
        return [];
    const parts = [];
    let current = "";
    let i = 0;
    while (i < trimmed.length) {
        const ch = trimmed[i];
        if (ch === "\\") {
            const next = trimmed[i + 1];
            if (next)
                current += next;
            i += 2;
            continue;
        }
        if (ch === ".") {
            if (current)
                parts.push(current);
            current = "";
            i += 1;
            continue;
        }
        if (ch === "[") {
            if (current)
                parts.push(current);
            current = "";
            const close = trimmed.indexOf("]", i);
            if (close === -1)
                throw new Error(`Invalid path (missing "]"): ${raw}`);
            const inside = trimmed.slice(i + 1, close).trim();
            if (!inside)
                throw new Error(`Invalid path (empty "[]"): ${raw}`);
            parts.push(inside);
            i = close + 1;
            continue;
        }
        current += ch;
        i += 1;
    }
    if (current)
        parts.push(current);
    return parts.map((part) => part.trim()).filter(Boolean);
}
function parseValue(raw, opts) {
    const trimmed = raw.trim();
    if (opts.json) {
        try {
            return JSON5.parse(trimmed);
        }
        catch (err) {
            throw new Error(`Failed to parse JSON5 value: ${String(err)}`);
        }
    }
    try {
        return JSON5.parse(trimmed);
    }
    catch {
        return raw;
    }
}
function getAtPath(root, path) {
    let current = root;
    for (const segment of path) {
        if (!current || typeof current !== "object")
            return { found: false };
        if (Array.isArray(current)) {
            if (!isIndexSegment(segment))
                return { found: false };
            const index = Number.parseInt(segment, 10);
            if (!Number.isFinite(index) || index < 0 || index >= current.length) {
                return { found: false };
            }
            current = current[index];
            continue;
        }
        const record = current;
        if (!(segment in record))
            return { found: false };
        current = record[segment];
    }
    return { found: true, value: current };
}
function setAtPath(root, path, value) {
    let current = root;
    for (let i = 0; i < path.length - 1; i += 1) {
        const segment = path[i];
        const next = path[i + 1];
        const nextIsIndex = Boolean(next && isIndexSegment(next));
        if (Array.isArray(current)) {
            if (!isIndexSegment(segment)) {
                throw new Error(`Expected numeric index for array segment "${segment}"`);
            }
            const index = Number.parseInt(segment, 10);
            const existing = current[index];
            if (!existing || typeof existing !== "object") {
                current[index] = nextIsIndex ? [] : {};
            }
            current = current[index];
            continue;
        }
        if (!current || typeof current !== "object") {
            throw new Error(`Cannot traverse into "${segment}" (not an object)`);
        }
        const record = current;
        const existing = record[segment];
        if (!existing || typeof existing !== "object") {
            record[segment] = nextIsIndex ? [] : {};
        }
        current = record[segment];
    }
    const last = path[path.length - 1];
    if (Array.isArray(current)) {
        if (!isIndexSegment(last)) {
            throw new Error(`Expected numeric index for array segment "${last}"`);
        }
        const index = Number.parseInt(last, 10);
        current[index] = value;
        return;
    }
    if (!current || typeof current !== "object") {
        throw new Error(`Cannot set "${last}" (parent is not an object)`);
    }
    current[last] = value;
}
function unsetAtPath(root, path) {
    let current = root;
    for (let i = 0; i < path.length - 1; i += 1) {
        const segment = path[i];
        if (!current || typeof current !== "object")
            return false;
        if (Array.isArray(current)) {
            if (!isIndexSegment(segment))
                return false;
            const index = Number.parseInt(segment, 10);
            if (!Number.isFinite(index) || index < 0 || index >= current.length)
                return false;
            current = current[index];
            continue;
        }
        const record = current;
        if (!(segment in record))
            return false;
        current = record[segment];
    }
    const last = path[path.length - 1];
    if (Array.isArray(current)) {
        if (!isIndexSegment(last))
            return false;
        const index = Number.parseInt(last, 10);
        if (!Number.isFinite(index) || index < 0 || index >= current.length)
            return false;
        current.splice(index, 1);
        return true;
    }
    if (!current || typeof current !== "object")
        return false;
    const record = current;
    if (!(last in record))
        return false;
    delete record[last];
    return true;
}
async function loadValidConfig() {
    const snapshot = await readConfigFileSnapshot();
    if (snapshot.valid)
        return snapshot;
    defaultRuntime.error(`Config invalid at ${shortenHomePath(snapshot.path)}.`);
    for (const issue of snapshot.issues) {
        defaultRuntime.error(`- ${issue.path || "<root>"}: ${issue.message}`);
    }
    defaultRuntime.error(`Run \`${formatCliCommand("clawdbot doctor")}\` to repair, then retry.`);
    defaultRuntime.exit(1);
    return snapshot;
}
export function registerConfigCli(program) {
    const cmd = program
        .command("config")
        .description("Config helpers (get/set/unset). Run without subcommand for the wizard.")
        .addHelpText("after", () => `\n${theme.muted("Docs:")} ${formatDocsLink("/cli/config", "docs.clawd.bot/cli/config")}\n`)
        .option("--section <section>", "Configure wizard sections (repeatable). Use with no subcommand.", (value, previous) => [...previous, value], [])
        .action(async (opts) => {
        const { CONFIGURE_WIZARD_SECTIONS, configureCommand, configureCommandWithSections } = await import("../commands/configure.js");
        const sections = Array.isArray(opts.section)
            ? opts.section
                .map((value) => (typeof value === "string" ? value.trim() : ""))
                .filter(Boolean)
            : [];
        if (sections.length === 0) {
            await configureCommand(defaultRuntime);
            return;
        }
        const invalid = sections.filter((s) => !CONFIGURE_WIZARD_SECTIONS.includes(s));
        if (invalid.length > 0) {
            defaultRuntime.error(`Invalid --section: ${invalid.join(", ")}. Expected one of: ${CONFIGURE_WIZARD_SECTIONS.join(", ")}.`);
            defaultRuntime.exit(1);
            return;
        }
        await configureCommandWithSections(sections, defaultRuntime);
    });
    cmd
        .command("get")
        .description("Get a config value by dot path")
        .argument("<path>", "Config path (dot or bracket notation)")
        .option("--json", "Output JSON", false)
        .action(async (path, opts) => {
        try {
            const parsedPath = parsePath(path);
            if (parsedPath.length === 0) {
                throw new Error("Path is empty.");
            }
            const snapshot = await loadValidConfig();
            const res = getAtPath(snapshot.config, parsedPath);
            if (!res.found) {
                defaultRuntime.error(danger(`Config path not found: ${path}`));
                defaultRuntime.exit(1);
                return;
            }
            if (opts.json) {
                defaultRuntime.log(JSON.stringify(res.value ?? null, null, 2));
                return;
            }
            if (typeof res.value === "string" ||
                typeof res.value === "number" ||
                typeof res.value === "boolean") {
                defaultRuntime.log(String(res.value));
                return;
            }
            defaultRuntime.log(JSON.stringify(res.value ?? null, null, 2));
        }
        catch (err) {
            defaultRuntime.error(danger(String(err)));
            defaultRuntime.exit(1);
        }
    });
    cmd
        .command("set")
        .description("Set a config value by dot path")
        .argument("<path>", "Config path (dot or bracket notation)")
        .argument("<value>", "Value (JSON5 or raw string)")
        .option("--json", "Parse value as JSON5 (required)", false)
        .action(async (path, value, opts) => {
        try {
            const parsedPath = parsePath(path);
            if (parsedPath.length === 0)
                throw new Error("Path is empty.");
            const parsedValue = parseValue(value, opts);
            const snapshot = await loadValidConfig();
            const next = snapshot.config;
            setAtPath(next, parsedPath, parsedValue);
            await writeConfigFile(next);
            defaultRuntime.log(info(`Updated ${path}. Restart the gateway to apply.`));
        }
        catch (err) {
            defaultRuntime.error(danger(String(err)));
            defaultRuntime.exit(1);
        }
    });
    cmd
        .command("unset")
        .description("Remove a config value by dot path")
        .argument("<path>", "Config path (dot or bracket notation)")
        .action(async (path) => {
        try {
            const parsedPath = parsePath(path);
            if (parsedPath.length === 0)
                throw new Error("Path is empty.");
            const snapshot = await loadValidConfig();
            const next = snapshot.config;
            const removed = unsetAtPath(next, parsedPath);
            if (!removed) {
                defaultRuntime.error(danger(`Config path not found: ${path}`));
                defaultRuntime.exit(1);
                return;
            }
            await writeConfigFile(next);
            defaultRuntime.log(info(`Removed ${path}. Restart the gateway to apply.`));
        }
        catch (err) {
            defaultRuntime.error(danger(String(err)));
            defaultRuntime.exit(1);
        }
    });
}
