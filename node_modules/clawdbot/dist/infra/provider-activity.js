const activity = new Map();
function keyFor(provider, accountId) {
    return `${provider}:${accountId || "default"}`;
}
function ensureEntry(provider, accountId) {
    const key = keyFor(provider, accountId);
    const existing = activity.get(key);
    if (existing)
        return existing;
    const created = { inboundAt: null, outboundAt: null };
    activity.set(key, created);
    return created;
}
export function recordProviderActivity(params) {
    const at = typeof params.at === "number" ? params.at : Date.now();
    const accountId = params.accountId?.trim() || "default";
    const entry = ensureEntry(params.provider, accountId);
    if (params.direction === "inbound")
        entry.inboundAt = at;
    if (params.direction === "outbound")
        entry.outboundAt = at;
}
export function getProviderActivity(params) {
    const accountId = params.accountId?.trim() || "default";
    return (activity.get(keyFor(params.provider, accountId)) ?? {
        inboundAt: null,
        outboundAt: null,
    });
}
export function resetProviderActivityForTest() {
    activity.clear();
}
