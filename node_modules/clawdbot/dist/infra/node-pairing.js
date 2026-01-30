import { randomUUID } from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import { resolveStateDir } from "../config/paths.js";
const PENDING_TTL_MS = 5 * 60 * 1000;
function resolvePaths(baseDir) {
    const root = baseDir ?? resolveStateDir();
    const dir = path.join(root, "nodes");
    return {
        dir,
        pendingPath: path.join(dir, "pending.json"),
        pairedPath: path.join(dir, "paired.json"),
    };
}
async function readJSON(filePath) {
    try {
        const raw = await fs.readFile(filePath, "utf8");
        return JSON.parse(raw);
    }
    catch {
        return null;
    }
}
async function writeJSONAtomic(filePath, value) {
    const dir = path.dirname(filePath);
    await fs.mkdir(dir, { recursive: true });
    const tmp = `${filePath}.${randomUUID()}.tmp`;
    await fs.writeFile(tmp, JSON.stringify(value, null, 2), "utf8");
    try {
        await fs.chmod(tmp, 0o600);
    }
    catch {
        // best-effort; ignore on platforms without chmod
    }
    await fs.rename(tmp, filePath);
    try {
        await fs.chmod(filePath, 0o600);
    }
    catch {
        // best-effort; ignore on platforms without chmod
    }
}
function pruneExpiredPending(pendingById, nowMs) {
    for (const [id, req] of Object.entries(pendingById)) {
        if (nowMs - req.ts > PENDING_TTL_MS) {
            delete pendingById[id];
        }
    }
}
let lock = Promise.resolve();
async function withLock(fn) {
    const prev = lock;
    let release;
    lock = new Promise((resolve) => {
        release = resolve;
    });
    await prev;
    try {
        return await fn();
    }
    finally {
        release?.();
    }
}
async function loadState(baseDir) {
    const { pendingPath, pairedPath } = resolvePaths(baseDir);
    const [pending, paired] = await Promise.all([
        readJSON(pendingPath),
        readJSON(pairedPath),
    ]);
    const state = {
        pendingById: pending ?? {},
        pairedByNodeId: paired ?? {},
    };
    pruneExpiredPending(state.pendingById, Date.now());
    return state;
}
async function persistState(state, baseDir) {
    const { pendingPath, pairedPath } = resolvePaths(baseDir);
    await Promise.all([
        writeJSONAtomic(pendingPath, state.pendingById),
        writeJSONAtomic(pairedPath, state.pairedByNodeId),
    ]);
}
function normalizeNodeId(nodeId) {
    return nodeId.trim();
}
function newToken() {
    return randomUUID().replaceAll("-", "");
}
export async function listNodePairing(baseDir) {
    const state = await loadState(baseDir);
    const pending = Object.values(state.pendingById).sort((a, b) => b.ts - a.ts);
    const paired = Object.values(state.pairedByNodeId).sort((a, b) => b.approvedAtMs - a.approvedAtMs);
    return { pending, paired };
}
export async function getPairedNode(nodeId, baseDir) {
    const state = await loadState(baseDir);
    return state.pairedByNodeId[normalizeNodeId(nodeId)] ?? null;
}
export async function requestNodePairing(req, baseDir) {
    return await withLock(async () => {
        const state = await loadState(baseDir);
        const nodeId = normalizeNodeId(req.nodeId);
        if (!nodeId) {
            throw new Error("nodeId required");
        }
        const existing = Object.values(state.pendingById).find((p) => p.nodeId === nodeId);
        if (existing) {
            return { status: "pending", request: existing, created: false };
        }
        const isRepair = Boolean(state.pairedByNodeId[nodeId]);
        const request = {
            requestId: randomUUID(),
            nodeId,
            displayName: req.displayName,
            platform: req.platform,
            version: req.version,
            coreVersion: req.coreVersion,
            uiVersion: req.uiVersion,
            deviceFamily: req.deviceFamily,
            modelIdentifier: req.modelIdentifier,
            caps: req.caps,
            commands: req.commands,
            permissions: req.permissions,
            remoteIp: req.remoteIp,
            silent: req.silent,
            isRepair,
            ts: Date.now(),
        };
        state.pendingById[request.requestId] = request;
        await persistState(state, baseDir);
        return { status: "pending", request, created: true };
    });
}
export async function approveNodePairing(requestId, baseDir) {
    return await withLock(async () => {
        const state = await loadState(baseDir);
        const pending = state.pendingById[requestId];
        if (!pending)
            return null;
        const now = Date.now();
        const existing = state.pairedByNodeId[pending.nodeId];
        const node = {
            nodeId: pending.nodeId,
            token: newToken(),
            displayName: pending.displayName,
            platform: pending.platform,
            version: pending.version,
            coreVersion: pending.coreVersion,
            uiVersion: pending.uiVersion,
            deviceFamily: pending.deviceFamily,
            modelIdentifier: pending.modelIdentifier,
            caps: pending.caps,
            commands: pending.commands,
            permissions: pending.permissions,
            remoteIp: pending.remoteIp,
            createdAtMs: existing?.createdAtMs ?? now,
            approvedAtMs: now,
        };
        delete state.pendingById[requestId];
        state.pairedByNodeId[pending.nodeId] = node;
        await persistState(state, baseDir);
        return { requestId, node };
    });
}
export async function rejectNodePairing(requestId, baseDir) {
    return await withLock(async () => {
        const state = await loadState(baseDir);
        const pending = state.pendingById[requestId];
        if (!pending)
            return null;
        delete state.pendingById[requestId];
        await persistState(state, baseDir);
        return { requestId, nodeId: pending.nodeId };
    });
}
export async function verifyNodeToken(nodeId, token, baseDir) {
    const state = await loadState(baseDir);
    const normalized = normalizeNodeId(nodeId);
    const node = state.pairedByNodeId[normalized];
    if (!node)
        return { ok: false };
    return node.token === token ? { ok: true, node } : { ok: false };
}
export async function updatePairedNodeMetadata(nodeId, patch, baseDir) {
    await withLock(async () => {
        const state = await loadState(baseDir);
        const normalized = normalizeNodeId(nodeId);
        const existing = state.pairedByNodeId[normalized];
        if (!existing)
            return;
        const next = {
            ...existing,
            displayName: patch.displayName ?? existing.displayName,
            platform: patch.platform ?? existing.platform,
            version: patch.version ?? existing.version,
            coreVersion: patch.coreVersion ?? existing.coreVersion,
            uiVersion: patch.uiVersion ?? existing.uiVersion,
            deviceFamily: patch.deviceFamily ?? existing.deviceFamily,
            modelIdentifier: patch.modelIdentifier ?? existing.modelIdentifier,
            remoteIp: patch.remoteIp ?? existing.remoteIp,
            caps: patch.caps ?? existing.caps,
            commands: patch.commands ?? existing.commands,
            bins: patch.bins ?? existing.bins,
            permissions: patch.permissions ?? existing.permissions,
            lastConnectedAtMs: patch.lastConnectedAtMs ?? existing.lastConnectedAtMs,
        };
        state.pairedByNodeId[normalized] = next;
        await persistState(state, baseDir);
    });
}
export async function renamePairedNode(nodeId, displayName, baseDir) {
    return await withLock(async () => {
        const state = await loadState(baseDir);
        const normalized = normalizeNodeId(nodeId);
        const existing = state.pairedByNodeId[normalized];
        if (!existing)
            return null;
        const trimmed = displayName.trim();
        if (!trimmed)
            throw new Error("displayName required");
        const next = { ...existing, displayName: trimmed };
        state.pairedByNodeId[normalized] = next;
        await persistState(state, baseDir);
        return next;
    });
}
