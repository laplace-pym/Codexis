import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
const UNICODE_SPACES = /[\u00A0\u2000-\u200A\u202F\u205F\u3000]/g;
function normalizeUnicodeSpaces(str) {
    return str.replace(UNICODE_SPACES, " ");
}
function expandPath(filePath) {
    const normalized = normalizeUnicodeSpaces(filePath);
    if (normalized === "~") {
        return os.homedir();
    }
    if (normalized.startsWith("~/")) {
        return os.homedir() + normalized.slice(1);
    }
    return normalized;
}
function resolveToCwd(filePath, cwd) {
    const expanded = expandPath(filePath);
    if (path.isAbsolute(expanded))
        return expanded;
    return path.resolve(cwd, expanded);
}
export function resolveSandboxPath(params) {
    const resolved = resolveToCwd(params.filePath, params.cwd);
    const rootResolved = path.resolve(params.root);
    const relative = path.relative(rootResolved, resolved);
    if (!relative || relative === "") {
        return { resolved, relative: "" };
    }
    if (relative.startsWith("..") || path.isAbsolute(relative)) {
        throw new Error(`Path escapes sandbox root (${shortPath(rootResolved)}): ${params.filePath}`);
    }
    return { resolved, relative };
}
export async function assertSandboxPath(params) {
    const resolved = resolveSandboxPath(params);
    await assertNoSymlink(resolved.relative, path.resolve(params.root));
    return resolved;
}
async function assertNoSymlink(relative, root) {
    if (!relative)
        return;
    const parts = relative.split(path.sep).filter(Boolean);
    let current = root;
    for (const part of parts) {
        current = path.join(current, part);
        try {
            const stat = await fs.lstat(current);
            if (stat.isSymbolicLink()) {
                throw new Error(`Symlink not allowed in sandbox path: ${current}`);
            }
        }
        catch (err) {
            const anyErr = err;
            if (anyErr.code === "ENOENT") {
                return;
            }
            throw err;
        }
    }
}
function shortPath(value) {
    if (value.startsWith(os.homedir())) {
        return `~${value.slice(os.homedir().length)}`;
    }
    return value;
}
