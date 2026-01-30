function isPlainObject(value) {
    return typeof value === "object" && value !== null && !Array.isArray(value);
}
export function applyMergePatch(base, patch) {
    if (!isPlainObject(patch)) {
        return patch;
    }
    const result = isPlainObject(base) ? { ...base } : {};
    for (const [key, value] of Object.entries(patch)) {
        if (value === null) {
            delete result[key];
            continue;
        }
        if (isPlainObject(value)) {
            const baseValue = result[key];
            result[key] = applyMergePatch(isPlainObject(baseValue) ? baseValue : {}, value);
            continue;
        }
        result[key] = value;
    }
    return result;
}
