export function normalizeAccountId(value) {
    if (typeof value !== "string")
        return undefined;
    const trimmed = value.trim();
    return trimmed || undefined;
}
