import fs from "node:fs/promises";
export async function safeStat(targetPath) {
    try {
        const lst = await fs.lstat(targetPath);
        return {
            ok: true,
            isSymlink: lst.isSymbolicLink(),
            isDir: lst.isDirectory(),
            mode: typeof lst.mode === "number" ? lst.mode : null,
            uid: typeof lst.uid === "number" ? lst.uid : null,
            gid: typeof lst.gid === "number" ? lst.gid : null,
        };
    }
    catch (err) {
        return {
            ok: false,
            isSymlink: false,
            isDir: false,
            mode: null,
            uid: null,
            gid: null,
            error: String(err),
        };
    }
}
export function modeBits(mode) {
    if (mode == null)
        return null;
    return mode & 0o777;
}
export function formatOctal(bits) {
    if (bits == null)
        return "unknown";
    return bits.toString(8).padStart(3, "0");
}
export function isWorldWritable(bits) {
    if (bits == null)
        return false;
    return (bits & 0o002) !== 0;
}
export function isGroupWritable(bits) {
    if (bits == null)
        return false;
    return (bits & 0o020) !== 0;
}
export function isWorldReadable(bits) {
    if (bits == null)
        return false;
    return (bits & 0o004) !== 0;
}
export function isGroupReadable(bits) {
    if (bits == null)
        return false;
    return (bits & 0o040) !== 0;
}
