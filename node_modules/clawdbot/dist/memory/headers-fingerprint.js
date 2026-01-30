function normalizeHeaderName(name) {
    return name.trim().toLowerCase();
}
export function fingerprintHeaderNames(headers) {
    if (!headers)
        return [];
    const out = [];
    for (const key of Object.keys(headers)) {
        const normalized = normalizeHeaderName(key);
        if (!normalized)
            continue;
        out.push(normalized);
    }
    out.sort((a, b) => a.localeCompare(b));
    return out;
}
