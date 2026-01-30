export function readString(meta, keys) {
    if (!meta)
        return undefined;
    for (const key of keys) {
        const value = meta[key];
        if (typeof value === "string" && value.trim())
            return value.trim();
    }
    return undefined;
}
export function readBool(meta, keys) {
    if (!meta)
        return undefined;
    for (const key of keys) {
        const value = meta[key];
        if (typeof value === "boolean")
            return value;
    }
    return undefined;
}
export function readNumber(meta, keys) {
    if (!meta)
        return undefined;
    for (const key of keys) {
        const value = meta[key];
        if (typeof value === "number" && Number.isFinite(value))
            return value;
    }
    return undefined;
}
