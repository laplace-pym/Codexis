export function asString(value) {
    return typeof value === "string" && value.trim().length > 0
        ? value.trim()
        : undefined;
}
export function isRecord(value) {
    return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}
