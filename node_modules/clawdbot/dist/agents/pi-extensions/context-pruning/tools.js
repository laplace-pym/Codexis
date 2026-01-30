function normalizePatterns(patterns) {
    if (!Array.isArray(patterns))
        return [];
    return patterns
        .map((p) => String(p ?? "")
        .trim()
        .toLowerCase())
        .filter(Boolean);
}
function compilePattern(pattern) {
    if (pattern === "*")
        return { kind: "all" };
    if (!pattern.includes("*"))
        return { kind: "exact", value: pattern };
    const escaped = pattern.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const re = new RegExp(`^${escaped.replaceAll("\\*", ".*")}$`);
    return { kind: "regex", value: re };
}
function compilePatterns(patterns) {
    return normalizePatterns(patterns).map(compilePattern);
}
function matchesAny(toolName, patterns) {
    for (const p of patterns) {
        if (p.kind === "all")
            return true;
        if (p.kind === "exact" && toolName === p.value)
            return true;
        if (p.kind === "regex" && p.value.test(toolName))
            return true;
    }
    return false;
}
export function makeToolPrunablePredicate(match) {
    const deny = compilePatterns(match.deny);
    const allow = compilePatterns(match.allow);
    return (toolName) => {
        const normalized = toolName.trim().toLowerCase();
        if (matchesAny(normalized, deny))
            return false;
        if (allow.length === 0)
            return true;
        return matchesAny(normalized, allow);
    };
}
