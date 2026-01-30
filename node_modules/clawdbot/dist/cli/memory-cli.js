import fsSync from "node:fs";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { resolveDefaultAgentId } from "../agents/agent-scope.js";
import { loadConfig } from "../config/config.js";
import { resolveSessionTranscriptsDirForAgent } from "../config/sessions/paths.js";
import { setVerbose } from "../globals.js";
import { withProgress, withProgressTotals } from "./progress.js";
import { formatErrorMessage, withManager } from "./cli-utils.js";
import { getMemorySearchManager } from "../memory/index.js";
import { listMemoryFiles } from "../memory/internal.js";
import { defaultRuntime } from "../runtime.js";
import { formatDocsLink } from "../terminal/links.js";
import { colorize, isRich, theme } from "../terminal/theme.js";
import { resolveStateDir } from "../config/paths.js";
import { shortenHomeInString, shortenHomePath } from "../utils.js";
function formatSourceLabel(source, workspaceDir, agentId) {
    if (source === "memory") {
        return shortenHomeInString(`memory (MEMORY.md + ${path.join(workspaceDir, "memory")}${path.sep}*.md)`);
    }
    if (source === "sessions") {
        const stateDir = resolveStateDir(process.env, os.homedir);
        return shortenHomeInString(`sessions (${path.join(stateDir, "agents", agentId, "sessions")}${path.sep}*.jsonl)`);
    }
    return source;
}
function resolveAgent(cfg, agent) {
    const trimmed = agent?.trim();
    if (trimmed)
        return trimmed;
    return resolveDefaultAgentId(cfg);
}
function resolveAgentIds(cfg, agent) {
    const trimmed = agent?.trim();
    if (trimmed)
        return [trimmed];
    const list = cfg.agents?.list ?? [];
    if (list.length > 0) {
        return list.map((entry) => entry.id).filter(Boolean);
    }
    return [resolveDefaultAgentId(cfg)];
}
async function checkReadableFile(pathname) {
    try {
        await fs.access(pathname, fsSync.constants.R_OK);
        return { exists: true };
    }
    catch (err) {
        const code = err.code;
        if (code === "ENOENT")
            return { exists: false };
        return {
            exists: true,
            issue: `${shortenHomePath(pathname)} not readable (${code ?? "error"})`,
        };
    }
}
async function scanSessionFiles(agentId) {
    const issues = [];
    const sessionsDir = resolveSessionTranscriptsDirForAgent(agentId);
    try {
        const entries = await fs.readdir(sessionsDir, { withFileTypes: true });
        const totalFiles = entries.filter((entry) => entry.isFile() && entry.name.endsWith(".jsonl")).length;
        return { source: "sessions", totalFiles, issues };
    }
    catch (err) {
        const code = err.code;
        if (code === "ENOENT") {
            issues.push(`sessions directory missing (${shortenHomePath(sessionsDir)})`);
            return { source: "sessions", totalFiles: 0, issues };
        }
        issues.push(`sessions directory not accessible (${shortenHomePath(sessionsDir)}): ${code ?? "error"}`);
        return { source: "sessions", totalFiles: null, issues };
    }
}
async function scanMemoryFiles(workspaceDir) {
    const issues = [];
    const memoryFile = path.join(workspaceDir, "MEMORY.md");
    const altMemoryFile = path.join(workspaceDir, "memory.md");
    const memoryDir = path.join(workspaceDir, "memory");
    const primary = await checkReadableFile(memoryFile);
    const alt = await checkReadableFile(altMemoryFile);
    if (primary.issue)
        issues.push(primary.issue);
    if (alt.issue)
        issues.push(alt.issue);
    let dirReadable = null;
    try {
        await fs.access(memoryDir, fsSync.constants.R_OK);
        dirReadable = true;
    }
    catch (err) {
        const code = err.code;
        if (code === "ENOENT") {
            issues.push(`memory directory missing (${shortenHomePath(memoryDir)})`);
            dirReadable = false;
        }
        else {
            issues.push(`memory directory not accessible (${shortenHomePath(memoryDir)}): ${code ?? "error"}`);
            dirReadable = null;
        }
    }
    let listed = [];
    let listedOk = false;
    try {
        listed = await listMemoryFiles(workspaceDir);
        listedOk = true;
    }
    catch (err) {
        const code = err.code;
        if (dirReadable !== null) {
            issues.push(`memory directory scan failed (${shortenHomePath(memoryDir)}): ${code ?? "error"}`);
            dirReadable = null;
        }
    }
    let totalFiles = 0;
    if (dirReadable === null) {
        totalFiles = null;
    }
    else {
        const files = new Set(listedOk ? listed : []);
        if (!listedOk) {
            if (primary.exists)
                files.add(memoryFile);
            if (alt.exists)
                files.add(altMemoryFile);
        }
        totalFiles = files.size;
    }
    if ((totalFiles ?? 0) === 0 && issues.length === 0) {
        issues.push(`no memory files found in ${shortenHomePath(workspaceDir)}`);
    }
    return { source: "memory", totalFiles, issues };
}
async function scanMemorySources(params) {
    const scans = [];
    for (const source of params.sources) {
        if (source === "memory") {
            scans.push(await scanMemoryFiles(params.workspaceDir));
        }
        if (source === "sessions") {
            scans.push(await scanSessionFiles(params.agentId));
        }
    }
    const issues = scans.flatMap((scan) => scan.issues);
    const totals = scans.map((scan) => scan.totalFiles);
    const numericTotals = totals.filter((total) => total !== null);
    const totalFiles = totals.some((total) => total === null)
        ? null
        : numericTotals.reduce((sum, total) => sum + total, 0);
    return { sources: scans, totalFiles, issues };
}
export async function runMemoryStatus(opts) {
    setVerbose(Boolean(opts.verbose));
    const cfg = loadConfig();
    const agentIds = resolveAgentIds(cfg, opts.agent);
    const allResults = [];
    for (const agentId of agentIds) {
        await withManager({
            getManager: () => getMemorySearchManager({ cfg, agentId }),
            onMissing: (error) => defaultRuntime.log(error ?? "Memory search disabled."),
            onCloseError: (err) => defaultRuntime.error(`Memory manager close failed: ${formatErrorMessage(err)}`),
            close: (manager) => manager.close(),
            run: async (manager) => {
                const deep = Boolean(opts.deep || opts.index);
                let embeddingProbe;
                let indexError;
                if (deep) {
                    await withProgress({ label: "Checking memory…", total: 2 }, async (progress) => {
                        progress.setLabel("Probing vector…");
                        await manager.probeVectorAvailability();
                        progress.tick();
                        progress.setLabel("Probing embeddings…");
                        embeddingProbe = await manager.probeEmbeddingAvailability();
                        progress.tick();
                    });
                    if (opts.index) {
                        await withProgressTotals({
                            label: "Indexing memory…",
                            total: 0,
                            fallback: opts.verbose ? "line" : undefined,
                        }, async (update, progress) => {
                            try {
                                await manager.sync({
                                    reason: "cli",
                                    progress: (syncUpdate) => {
                                        update({
                                            completed: syncUpdate.completed,
                                            total: syncUpdate.total,
                                            label: syncUpdate.label,
                                        });
                                        if (syncUpdate.label)
                                            progress.setLabel(syncUpdate.label);
                                    },
                                });
                            }
                            catch (err) {
                                indexError = formatErrorMessage(err);
                                defaultRuntime.error(`Memory index failed: ${indexError}`);
                                process.exitCode = 1;
                            }
                        });
                    }
                }
                else {
                    await manager.probeVectorAvailability();
                }
                const status = manager.status();
                const sources = (status.sources?.length ? status.sources : ["memory"]);
                const scan = await scanMemorySources({
                    workspaceDir: status.workspaceDir,
                    agentId,
                    sources,
                });
                allResults.push({ agentId, status, embeddingProbe, indexError, scan });
            },
        });
    }
    if (opts.json) {
        defaultRuntime.log(JSON.stringify(allResults, null, 2));
        return;
    }
    const rich = isRich();
    const heading = (text) => colorize(rich, theme.heading, text);
    const muted = (text) => colorize(rich, theme.muted, text);
    const info = (text) => colorize(rich, theme.info, text);
    const success = (text) => colorize(rich, theme.success, text);
    const warn = (text) => colorize(rich, theme.warn, text);
    const accent = (text) => colorize(rich, theme.accent, text);
    const label = (text) => muted(`${text}:`);
    for (const result of allResults) {
        const { agentId, status, embeddingProbe, indexError, scan } = result;
        const totalFiles = scan?.totalFiles ?? null;
        const indexedLabel = totalFiles === null
            ? `${status.files}/? files · ${status.chunks} chunks`
            : `${status.files}/${totalFiles} files · ${status.chunks} chunks`;
        if (opts.index) {
            const line = indexError ? `Memory index failed: ${indexError}` : "Memory index complete.";
            defaultRuntime.log(line);
        }
        const lines = [
            `${heading("Memory Search")} ${muted(`(${agentId})`)}`,
            `${label("Provider")} ${info(status.provider)} ${muted(`(requested: ${status.requestedProvider})`)}`,
            `${label("Model")} ${info(status.model)}`,
            status.sources?.length ? `${label("Sources")} ${info(status.sources.join(", "))}` : null,
            `${label("Indexed")} ${success(indexedLabel)}`,
            `${label("Dirty")} ${status.dirty ? warn("yes") : muted("no")}`,
            `${label("Store")} ${info(shortenHomePath(status.dbPath))}`,
            `${label("Workspace")} ${info(shortenHomePath(status.workspaceDir))}`,
        ].filter(Boolean);
        if (embeddingProbe) {
            const state = embeddingProbe.ok ? "ready" : "unavailable";
            const stateColor = embeddingProbe.ok ? theme.success : theme.warn;
            lines.push(`${label("Embeddings")} ${colorize(rich, stateColor, state)}`);
            if (embeddingProbe.error) {
                lines.push(`${label("Embeddings error")} ${warn(embeddingProbe.error)}`);
            }
        }
        if (status.sourceCounts?.length) {
            lines.push(label("By source"));
            for (const entry of status.sourceCounts) {
                const total = scan?.sources.find((scanEntry) => scanEntry.source === entry.source)?.totalFiles;
                const counts = total === null
                    ? `${entry.files}/? files · ${entry.chunks} chunks`
                    : `${entry.files}/${total} files · ${entry.chunks} chunks`;
                lines.push(`  ${accent(entry.source)} ${muted("·")} ${muted(counts)}`);
            }
        }
        if (status.fallback) {
            lines.push(`${label("Fallback")} ${warn(status.fallback.from)}`);
        }
        if (status.vector) {
            const vectorState = status.vector.enabled
                ? status.vector.available === undefined
                    ? "unknown"
                    : status.vector.available
                        ? "ready"
                        : "unavailable"
                : "disabled";
            const vectorColor = vectorState === "ready"
                ? theme.success
                : vectorState === "unavailable"
                    ? theme.warn
                    : theme.muted;
            lines.push(`${label("Vector")} ${colorize(rich, vectorColor, vectorState)}`);
            if (status.vector.dims) {
                lines.push(`${label("Vector dims")} ${info(String(status.vector.dims))}`);
            }
            if (status.vector.extensionPath) {
                lines.push(`${label("Vector path")} ${info(shortenHomePath(status.vector.extensionPath))}`);
            }
            if (status.vector.loadError) {
                lines.push(`${label("Vector error")} ${warn(status.vector.loadError)}`);
            }
        }
        if (status.fts) {
            const ftsState = status.fts.enabled
                ? status.fts.available
                    ? "ready"
                    : "unavailable"
                : "disabled";
            const ftsColor = ftsState === "ready"
                ? theme.success
                : ftsState === "unavailable"
                    ? theme.warn
                    : theme.muted;
            lines.push(`${label("FTS")} ${colorize(rich, ftsColor, ftsState)}`);
            if (status.fts.error) {
                lines.push(`${label("FTS error")} ${warn(status.fts.error)}`);
            }
        }
        if (status.cache) {
            const cacheState = status.cache.enabled ? "enabled" : "disabled";
            const cacheColor = status.cache.enabled ? theme.success : theme.muted;
            const suffix = status.cache.enabled && typeof status.cache.entries === "number"
                ? ` (${status.cache.entries} entries)`
                : "";
            lines.push(`${label("Embedding cache")} ${colorize(rich, cacheColor, cacheState)}${suffix}`);
            if (status.cache.enabled && typeof status.cache.maxEntries === "number") {
                lines.push(`${label("Cache cap")} ${info(String(status.cache.maxEntries))}`);
            }
        }
        if (status.batch) {
            const batchState = status.batch.enabled ? "enabled" : "disabled";
            const batchColor = status.batch.enabled ? theme.success : theme.warn;
            const batchSuffix = ` (failures ${status.batch.failures}/${status.batch.limit})`;
            lines.push(`${label("Batch")} ${colorize(rich, batchColor, batchState)}${muted(batchSuffix)}`);
            if (status.batch.lastError) {
                lines.push(`${label("Batch error")} ${warn(status.batch.lastError)}`);
            }
        }
        if (status.fallback?.reason) {
            lines.push(muted(status.fallback.reason));
        }
        if (indexError) {
            lines.push(`${label("Index error")} ${warn(indexError)}`);
        }
        if (scan?.issues.length) {
            lines.push(label("Issues"));
            for (const issue of scan.issues) {
                lines.push(`  ${warn(issue)}`);
            }
        }
        defaultRuntime.log(lines.join("\n"));
        defaultRuntime.log("");
    }
}
export function registerMemoryCli(program) {
    const memory = program
        .command("memory")
        .description("Memory search tools")
        .addHelpText("after", () => `\n${theme.muted("Docs:")} ${formatDocsLink("/cli/memory", "docs.clawd.bot/cli/memory")}\n`);
    memory
        .command("status")
        .description("Show memory search index status")
        .option("--agent <id>", "Agent id (default: default agent)")
        .option("--json", "Print JSON")
        .option("--deep", "Probe embedding provider availability")
        .option("--index", "Reindex if dirty (implies --deep)")
        .option("--verbose", "Verbose logging", false)
        .action(async (opts) => {
        await runMemoryStatus(opts);
    });
    memory
        .command("index")
        .description("Reindex memory files")
        .option("--agent <id>", "Agent id (default: default agent)")
        .option("--force", "Force full reindex", false)
        .option("--verbose", "Verbose logging", false)
        .action(async (opts) => {
        setVerbose(Boolean(opts.verbose));
        const cfg = loadConfig();
        const agentIds = resolveAgentIds(cfg, opts.agent);
        for (const agentId of agentIds) {
            await withManager({
                getManager: () => getMemorySearchManager({ cfg, agentId }),
                onMissing: (error) => defaultRuntime.log(error ?? "Memory search disabled."),
                onCloseError: (err) => defaultRuntime.error(`Memory manager close failed: ${formatErrorMessage(err)}`),
                close: (manager) => manager.close(),
                run: async (manager) => {
                    try {
                        if (opts.verbose) {
                            const status = manager.status();
                            const rich = isRich();
                            const heading = (text) => colorize(rich, theme.heading, text);
                            const muted = (text) => colorize(rich, theme.muted, text);
                            const info = (text) => colorize(rich, theme.info, text);
                            const warn = (text) => colorize(rich, theme.warn, text);
                            const label = (text) => muted(`${text}:`);
                            const sourceLabels = status.sources.map((source) => formatSourceLabel(source, status.workspaceDir, agentId));
                            const lines = [
                                `${heading("Memory Index")} ${muted(`(${agentId})`)}`,
                                `${label("Provider")} ${info(status.provider)} ${muted(`(requested: ${status.requestedProvider})`)}`,
                                `${label("Model")} ${info(status.model)}`,
                                sourceLabels.length
                                    ? `${label("Sources")} ${info(sourceLabels.join(", "))}`
                                    : null,
                            ].filter(Boolean);
                            if (status.fallback) {
                                lines.push(`${label("Fallback")} ${warn(status.fallback.from)}`);
                            }
                            defaultRuntime.log(lines.join("\n"));
                            defaultRuntime.log("");
                        }
                        const startedAt = Date.now();
                        let lastLabel = "Indexing memory…";
                        let lastCompleted = 0;
                        let lastTotal = 0;
                        const formatElapsed = () => {
                            const elapsedMs = Math.max(0, Date.now() - startedAt);
                            const seconds = Math.floor(elapsedMs / 1000);
                            const minutes = Math.floor(seconds / 60);
                            const remainingSeconds = seconds % 60;
                            return `${minutes}:${String(remainingSeconds).padStart(2, "0")}`;
                        };
                        const formatEta = () => {
                            if (lastTotal <= 0 || lastCompleted <= 0)
                                return null;
                            const elapsedMs = Math.max(1, Date.now() - startedAt);
                            const rate = lastCompleted / elapsedMs;
                            if (!Number.isFinite(rate) || rate <= 0)
                                return null;
                            const remainingMs = Math.max(0, (lastTotal - lastCompleted) / rate);
                            const seconds = Math.floor(remainingMs / 1000);
                            const minutes = Math.floor(seconds / 60);
                            const remainingSeconds = seconds % 60;
                            return `${minutes}:${String(remainingSeconds).padStart(2, "0")}`;
                        };
                        const buildLabel = () => {
                            const elapsed = formatElapsed();
                            const eta = formatEta();
                            return eta
                                ? `${lastLabel} · elapsed ${elapsed} · eta ${eta}`
                                : `${lastLabel} · elapsed ${elapsed}`;
                        };
                        await withProgressTotals({
                            label: "Indexing memory…",
                            total: 0,
                            fallback: opts.verbose ? "line" : undefined,
                        }, async (update, progress) => {
                            const interval = setInterval(() => {
                                progress.setLabel(buildLabel());
                            }, 1000);
                            try {
                                await manager.sync({
                                    reason: "cli",
                                    force: opts.force,
                                    progress: (syncUpdate) => {
                                        if (syncUpdate.label)
                                            lastLabel = syncUpdate.label;
                                        lastCompleted = syncUpdate.completed;
                                        lastTotal = syncUpdate.total;
                                        update({
                                            completed: syncUpdate.completed,
                                            total: syncUpdate.total,
                                            label: buildLabel(),
                                        });
                                        progress.setLabel(buildLabel());
                                    },
                                });
                            }
                            finally {
                                clearInterval(interval);
                            }
                        });
                        defaultRuntime.log(`Memory index updated (${agentId}).`);
                    }
                    catch (err) {
                        const message = formatErrorMessage(err);
                        defaultRuntime.error(`Memory index failed (${agentId}): ${message}`);
                        process.exitCode = 1;
                    }
                },
            });
        }
    });
    memory
        .command("search")
        .description("Search memory files")
        .argument("<query>", "Search query")
        .option("--agent <id>", "Agent id (default: default agent)")
        .option("--max-results <n>", "Max results", (value) => Number(value))
        .option("--min-score <n>", "Minimum score", (value) => Number(value))
        .option("--json", "Print JSON")
        .action(async (query, opts) => {
        const cfg = loadConfig();
        const agentId = resolveAgent(cfg, opts.agent);
        await withManager({
            getManager: () => getMemorySearchManager({ cfg, agentId }),
            onMissing: (error) => defaultRuntime.log(error ?? "Memory search disabled."),
            onCloseError: (err) => defaultRuntime.error(`Memory manager close failed: ${formatErrorMessage(err)}`),
            close: (manager) => manager.close(),
            run: async (manager) => {
                let results;
                try {
                    results = await manager.search(query, {
                        maxResults: opts.maxResults,
                        minScore: opts.minScore,
                    });
                }
                catch (err) {
                    const message = formatErrorMessage(err);
                    defaultRuntime.error(`Memory search failed: ${message}`);
                    process.exitCode = 1;
                    return;
                }
                if (opts.json) {
                    defaultRuntime.log(JSON.stringify({ results }, null, 2));
                    return;
                }
                if (results.length === 0) {
                    defaultRuntime.log("No matches.");
                    return;
                }
                const rich = isRich();
                const lines = [];
                for (const result of results) {
                    lines.push(`${colorize(rich, theme.success, result.score.toFixed(3))} ${colorize(rich, theme.accent, `${shortenHomePath(result.path)}:${result.startLine}-${result.endLine}`)}`);
                    lines.push(colorize(rich, theme.muted, result.snippet));
                    lines.push("");
                }
                defaultRuntime.log(lines.join("\n").trim());
            },
        });
    });
}
