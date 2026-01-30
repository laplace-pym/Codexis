import fs from "node:fs/promises";
import path from "node:path";
import { resolveDefaultAgentWorkspaceDir } from "../agents/workspace.js";
import { resolveHomeDir, resolveUserPath, shortenHomeInString } from "../utils.js";
export function collectWorkspaceDirs(cfg) {
    const dirs = new Set();
    const defaults = cfg?.agents?.defaults;
    if (typeof defaults?.workspace === "string" && defaults.workspace.trim()) {
        dirs.add(resolveUserPath(defaults.workspace));
    }
    const list = Array.isArray(cfg?.agents?.list) ? cfg?.agents?.list : [];
    for (const agent of list) {
        const workspace = agent.workspace;
        if (typeof workspace === "string" && workspace.trim()) {
            dirs.add(resolveUserPath(workspace));
        }
    }
    if (dirs.size === 0) {
        dirs.add(resolveDefaultAgentWorkspaceDir());
    }
    return [...dirs];
}
export function isPathWithin(child, parent) {
    const relative = path.relative(parent, child);
    return relative === "" || (!relative.startsWith("..") && !path.isAbsolute(relative));
}
function isUnsafeRemovalTarget(target) {
    if (!target.trim())
        return true;
    const resolved = path.resolve(target);
    const root = path.parse(resolved).root;
    if (resolved === root)
        return true;
    const home = resolveHomeDir();
    if (home && resolved === path.resolve(home))
        return true;
    return false;
}
export async function removePath(target, runtime, opts) {
    if (!target?.trim())
        return { ok: false, skipped: true };
    const resolved = path.resolve(target);
    const label = opts?.label ?? resolved;
    const displayLabel = shortenHomeInString(label);
    if (isUnsafeRemovalTarget(resolved)) {
        runtime.error(`Refusing to remove unsafe path: ${displayLabel}`);
        return { ok: false };
    }
    if (opts?.dryRun) {
        runtime.log(`[dry-run] remove ${displayLabel}`);
        return { ok: true, skipped: true };
    }
    try {
        await fs.rm(resolved, { recursive: true, force: true });
        runtime.log(`Removed ${displayLabel}`);
        return { ok: true };
    }
    catch (err) {
        runtime.error(`Failed to remove ${displayLabel}: ${String(err)}`);
        return { ok: false };
    }
}
export async function listAgentSessionDirs(stateDir) {
    const root = path.join(stateDir, "agents");
    try {
        const entries = await fs.readdir(root, { withFileTypes: true });
        return entries
            .filter((entry) => entry.isDirectory())
            .map((entry) => path.join(root, entry.name, "sessions"));
    }
    catch {
        return [];
    }
}
