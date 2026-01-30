export function asString(value) {
    return typeof value === "string" && value.trim().length > 0 ? value.trim() : undefined;
}
export function isRecord(value) {
    return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}
export function formatMatchMetadata(params) {
    const matchKey = typeof params.matchKey === "string"
        ? params.matchKey
        : typeof params.matchKey === "number"
            ? String(params.matchKey)
            : undefined;
    const matchSource = asString(params.matchSource);
    const parts = [
        matchKey ? `matchKey=${matchKey}` : null,
        matchSource ? `matchSource=${matchSource}` : null,
    ].filter((entry) => Boolean(entry));
    return parts.length > 0 ? parts.join(" ") : undefined;
}
export function appendMatchMetadata(message, params) {
    const meta = formatMatchMetadata(params);
    return meta ? `${message} (${meta})` : message;
}
