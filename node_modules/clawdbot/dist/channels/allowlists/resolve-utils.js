export function mergeAllowlist(params) {
    const seen = new Set();
    const merged = [];
    const push = (value) => {
        const normalized = value.trim();
        if (!normalized)
            return;
        const key = normalized.toLowerCase();
        if (seen.has(key))
            return;
        seen.add(key);
        merged.push(normalized);
    };
    for (const entry of params.existing ?? []) {
        push(String(entry));
    }
    for (const entry of params.additions) {
        push(entry);
    }
    return merged;
}
export function summarizeMapping(label, mapping, unresolved, runtime) {
    const lines = [];
    if (mapping.length > 0) {
        const sample = mapping.slice(0, 6);
        const suffix = mapping.length > sample.length ? ` (+${mapping.length - sample.length})` : "";
        lines.push(`${label} resolved: ${sample.join(", ")}${suffix}`);
    }
    if (unresolved.length > 0) {
        const sample = unresolved.slice(0, 6);
        const suffix = unresolved.length > sample.length ? ` (+${unresolved.length - sample.length})` : "";
        lines.push(`${label} unresolved: ${sample.join(", ")}${suffix}`);
    }
    if (lines.length > 0) {
        runtime.log?.(lines.join("\n"));
    }
}
