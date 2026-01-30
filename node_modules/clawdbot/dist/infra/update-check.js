import fs from "node:fs/promises";
import path from "node:path";
import { runCommandWithTimeout } from "../process/exec.js";
import { parseSemver } from "./runtime-guard.js";
import { channelToNpmTag } from "./update-channels.js";
async function exists(p) {
    try {
        await fs.access(p);
        return true;
    }
    catch {
        return false;
    }
}
async function detectPackageManager(root) {
    try {
        const raw = await fs.readFile(path.join(root, "package.json"), "utf-8");
        const parsed = JSON.parse(raw);
        const pm = parsed?.packageManager?.split("@")[0]?.trim();
        if (pm === "pnpm" || pm === "bun" || pm === "npm")
            return pm;
    }
    catch {
        // ignore
    }
    const files = await fs.readdir(root).catch(() => []);
    if (files.includes("pnpm-lock.yaml"))
        return "pnpm";
    if (files.includes("bun.lockb"))
        return "bun";
    if (files.includes("package-lock.json"))
        return "npm";
    return "unknown";
}
async function detectGitRoot(root) {
    const res = await runCommandWithTimeout(["git", "-C", root, "rev-parse", "--show-toplevel"], {
        timeoutMs: 4000,
    }).catch(() => null);
    if (!res || res.code !== 0)
        return null;
    const top = res.stdout.trim();
    return top ? path.resolve(top) : null;
}
export async function checkGitUpdateStatus(params) {
    const timeoutMs = params.timeoutMs ?? 6000;
    const root = path.resolve(params.root);
    const base = {
        root,
        sha: null,
        tag: null,
        branch: null,
        upstream: null,
        dirty: null,
        ahead: null,
        behind: null,
        fetchOk: null,
    };
    const branchRes = await runCommandWithTimeout(["git", "-C", root, "rev-parse", "--abbrev-ref", "HEAD"], { timeoutMs }).catch(() => null);
    if (!branchRes || branchRes.code !== 0) {
        return { ...base, error: branchRes?.stderr?.trim() || "git unavailable" };
    }
    const branch = branchRes.stdout.trim() || null;
    const shaRes = await runCommandWithTimeout(["git", "-C", root, "rev-parse", "HEAD"], {
        timeoutMs,
    }).catch(() => null);
    const sha = shaRes && shaRes.code === 0 ? shaRes.stdout.trim() : null;
    const tagRes = await runCommandWithTimeout(["git", "-C", root, "describe", "--tags", "--exact-match"], { timeoutMs }).catch(() => null);
    const tag = tagRes && tagRes.code === 0 ? tagRes.stdout.trim() : null;
    const upstreamRes = await runCommandWithTimeout(["git", "-C", root, "rev-parse", "--abbrev-ref", "@{upstream}"], { timeoutMs }).catch(() => null);
    const upstream = upstreamRes && upstreamRes.code === 0 ? upstreamRes.stdout.trim() : null;
    const dirtyRes = await runCommandWithTimeout(["git", "-C", root, "status", "--porcelain"], {
        timeoutMs,
    }).catch(() => null);
    const dirty = dirtyRes && dirtyRes.code === 0 ? dirtyRes.stdout.trim().length > 0 : null;
    const fetchOk = params.fetch
        ? await runCommandWithTimeout(["git", "-C", root, "fetch", "--quiet", "--prune"], { timeoutMs })
            .then((r) => r.code === 0)
            .catch(() => false)
        : null;
    const counts = upstream && upstream.length > 0
        ? await runCommandWithTimeout(["git", "-C", root, "rev-list", "--left-right", "--count", `HEAD...${upstream}`], { timeoutMs }).catch(() => null)
        : null;
    const parseCounts = (raw) => {
        const parts = raw.trim().split(/\s+/);
        if (parts.length < 2)
            return null;
        const ahead = Number.parseInt(parts[0] ?? "", 10);
        const behind = Number.parseInt(parts[1] ?? "", 10);
        if (!Number.isFinite(ahead) || !Number.isFinite(behind))
            return null;
        return { ahead, behind };
    };
    const parsed = counts && counts.code === 0 ? parseCounts(counts.stdout) : null;
    return {
        root,
        sha,
        tag,
        branch,
        upstream,
        dirty,
        ahead: parsed?.ahead ?? null,
        behind: parsed?.behind ?? null,
        fetchOk,
    };
}
async function statMtimeMs(p) {
    try {
        const st = await fs.stat(p);
        return st.mtimeMs;
    }
    catch {
        return null;
    }
}
function resolveDepsMarker(params) {
    const root = params.root;
    if (params.manager === "pnpm") {
        return {
            lockfilePath: path.join(root, "pnpm-lock.yaml"),
            markerPath: path.join(root, "node_modules", ".modules.yaml"),
        };
    }
    if (params.manager === "bun") {
        return {
            lockfilePath: path.join(root, "bun.lockb"),
            markerPath: path.join(root, "node_modules"),
        };
    }
    if (params.manager === "npm") {
        return {
            lockfilePath: path.join(root, "package-lock.json"),
            markerPath: path.join(root, "node_modules"),
        };
    }
    return { lockfilePath: null, markerPath: null };
}
export async function checkDepsStatus(params) {
    const root = path.resolve(params.root);
    const { lockfilePath, markerPath } = resolveDepsMarker({
        root,
        manager: params.manager,
    });
    if (!lockfilePath || !markerPath) {
        return {
            manager: params.manager,
            status: "unknown",
            lockfilePath,
            markerPath,
            reason: "unknown package manager",
        };
    }
    const lockExists = await exists(lockfilePath);
    const markerExists = await exists(markerPath);
    if (!lockExists) {
        return {
            manager: params.manager,
            status: "unknown",
            lockfilePath,
            markerPath,
            reason: "lockfile missing",
        };
    }
    if (!markerExists) {
        return {
            manager: params.manager,
            status: "missing",
            lockfilePath,
            markerPath,
            reason: "node_modules marker missing",
        };
    }
    const lockMtime = await statMtimeMs(lockfilePath);
    const markerMtime = await statMtimeMs(markerPath);
    if (!lockMtime || !markerMtime) {
        return {
            manager: params.manager,
            status: "unknown",
            lockfilePath,
            markerPath,
        };
    }
    if (lockMtime > markerMtime + 1000) {
        return {
            manager: params.manager,
            status: "stale",
            lockfilePath,
            markerPath,
            reason: "lockfile newer than install marker",
        };
    }
    return {
        manager: params.manager,
        status: "ok",
        lockfilePath,
        markerPath,
    };
}
async function fetchWithTimeout(url, timeoutMs) {
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), Math.max(250, timeoutMs));
    try {
        return await fetch(url, { signal: ctrl.signal });
    }
    finally {
        clearTimeout(t);
    }
}
export async function fetchNpmLatestVersion(params) {
    const res = await fetchNpmTagVersion({ tag: "latest", timeoutMs: params?.timeoutMs });
    return {
        latestVersion: res.version,
        error: res.error,
    };
}
export async function fetchNpmTagVersion(params) {
    const timeoutMs = params?.timeoutMs ?? 3500;
    const tag = params.tag;
    try {
        const res = await fetchWithTimeout(`https://registry.npmjs.org/clawdbot/${encodeURIComponent(tag)}`, timeoutMs);
        if (!res.ok) {
            return { tag, version: null, error: `HTTP ${res.status}` };
        }
        const json = (await res.json());
        const version = typeof json?.version === "string" ? json.version : null;
        return { tag, version };
    }
    catch (err) {
        return { tag, version: null, error: String(err) };
    }
}
export async function resolveNpmChannelTag(params) {
    const channelTag = channelToNpmTag(params.channel);
    const channelStatus = await fetchNpmTagVersion({ tag: channelTag, timeoutMs: params.timeoutMs });
    if (params.channel !== "beta") {
        return { tag: channelTag, version: channelStatus.version };
    }
    const latestStatus = await fetchNpmTagVersion({ tag: "latest", timeoutMs: params.timeoutMs });
    if (!latestStatus.version) {
        return { tag: channelTag, version: channelStatus.version };
    }
    if (!channelStatus.version) {
        return { tag: "latest", version: latestStatus.version };
    }
    const cmp = compareSemverStrings(channelStatus.version, latestStatus.version);
    if (cmp != null && cmp < 0) {
        return { tag: "latest", version: latestStatus.version };
    }
    return { tag: channelTag, version: channelStatus.version };
}
export function compareSemverStrings(a, b) {
    const pa = parseSemver(a);
    const pb = parseSemver(b);
    if (!pa || !pb)
        return null;
    if (pa.major !== pb.major)
        return pa.major < pb.major ? -1 : 1;
    if (pa.minor !== pb.minor)
        return pa.minor < pb.minor ? -1 : 1;
    if (pa.patch !== pb.patch)
        return pa.patch < pb.patch ? -1 : 1;
    return 0;
}
export async function checkUpdateStatus(params) {
    const timeoutMs = params.timeoutMs ?? 6000;
    const root = params.root ? path.resolve(params.root) : null;
    if (!root) {
        return {
            root: null,
            installKind: "unknown",
            packageManager: "unknown",
            registry: params.includeRegistry ? await fetchNpmLatestVersion({ timeoutMs }) : undefined,
        };
    }
    const pm = await detectPackageManager(root);
    const gitRoot = await detectGitRoot(root);
    const isGit = gitRoot && path.resolve(gitRoot) === root;
    const installKind = isGit ? "git" : "package";
    const git = isGit
        ? await checkGitUpdateStatus({
            root,
            timeoutMs,
            fetch: Boolean(params.fetchGit),
        })
        : undefined;
    const deps = await checkDepsStatus({ root, manager: pm });
    const registry = params.includeRegistry ? await fetchNpmLatestVersion({ timeoutMs }) : undefined;
    return {
        root,
        installKind,
        packageManager: pm,
        git,
        deps,
        registry,
    };
}
