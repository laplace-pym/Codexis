export function formatAge(msAgo) {
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
export function parsePairingList(value) {
    const obj = typeof value === "object" && value !== null ? value : {};
    const pending = Array.isArray(obj.pending) ? obj.pending : [];
    const paired = Array.isArray(obj.paired) ? obj.paired : [];
    return { pending, paired };
}
export function parseNodeList(value) {
    const obj = typeof value === "object" && value !== null ? value : {};
    return Array.isArray(obj.nodes) ? obj.nodes : [];
}
export function formatPermissions(raw) {
    if (!raw || typeof raw !== "object" || Array.isArray(raw))
        return null;
    const entries = Object.entries(raw)
        .map(([key, value]) => [String(key).trim(), value === true])
        .filter(([key]) => key.length > 0)
        .sort((a, b) => a[0].localeCompare(b[0]));
    if (entries.length === 0)
        return null;
    const parts = entries.map(([key, granted]) => `${key}=${granted ? "yes" : "no"}`);
    return `[${parts.join(", ")}]`;
}
