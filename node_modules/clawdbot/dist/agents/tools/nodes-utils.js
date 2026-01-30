import { callGatewayTool } from "./gateway.js";
function parseNodeList(value) {
    const obj = typeof value === "object" && value !== null ? value : {};
    return Array.isArray(obj.nodes) ? obj.nodes : [];
}
function parsePairingList(value) {
    const obj = typeof value === "object" && value !== null ? value : {};
    const pending = Array.isArray(obj.pending) ? obj.pending : [];
    const paired = Array.isArray(obj.paired) ? obj.paired : [];
    return { pending, paired };
}
function normalizeNodeKey(value) {
    return value
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, "-")
        .replace(/^-+/, "")
        .replace(/-+$/, "");
}
async function loadNodes(opts) {
    try {
        const res = (await callGatewayTool("node.list", opts, {}));
        return parseNodeList(res);
    }
    catch {
        const res = (await callGatewayTool("node.pair.list", opts, {}));
        const { paired } = parsePairingList(res);
        return paired.map((n) => ({
            nodeId: n.nodeId,
            displayName: n.displayName,
            platform: n.platform,
            remoteIp: n.remoteIp,
        }));
    }
}
function pickDefaultNode(nodes) {
    const withCanvas = nodes.filter((n) => Array.isArray(n.caps) ? n.caps.includes("canvas") : true);
    if (withCanvas.length === 0)
        return null;
    const connected = withCanvas.filter((n) => n.connected);
    const candidates = connected.length > 0 ? connected : withCanvas;
    if (candidates.length === 1)
        return candidates[0];
    const local = candidates.filter((n) => n.platform?.toLowerCase().startsWith("mac") &&
        typeof n.nodeId === "string" &&
        n.nodeId.startsWith("mac-"));
    if (local.length === 1)
        return local[0];
    return null;
}
export async function listNodes(opts) {
    return loadNodes(opts);
}
export function resolveNodeIdFromList(nodes, query, allowDefault = false) {
    const q = String(query ?? "").trim();
    if (!q) {
        if (allowDefault) {
            const picked = pickDefaultNode(nodes);
            if (picked)
                return picked.nodeId;
        }
        throw new Error("node required");
    }
    const qNorm = normalizeNodeKey(q);
    const matches = nodes.filter((n) => {
        if (n.nodeId === q)
            return true;
        if (typeof n.remoteIp === "string" && n.remoteIp === q)
            return true;
        const name = typeof n.displayName === "string" ? n.displayName : "";
        if (name && normalizeNodeKey(name) === qNorm)
            return true;
        if (q.length >= 6 && n.nodeId.startsWith(q))
            return true;
        return false;
    });
    if (matches.length === 1)
        return matches[0].nodeId;
    if (matches.length === 0) {
        const known = nodes
            .map((n) => n.displayName || n.remoteIp || n.nodeId)
            .filter(Boolean)
            .join(", ");
        throw new Error(`unknown node: ${q}${known ? ` (known: ${known})` : ""}`);
    }
    throw new Error(`ambiguous node: ${q} (matches: ${matches
        .map((n) => n.displayName || n.remoteIp || n.nodeId)
        .join(", ")})`);
}
export async function resolveNodeId(opts, query, allowDefault = false) {
    const nodes = await loadNodes(opts);
    return resolveNodeIdFromList(nodes, query, allowDefault);
}
