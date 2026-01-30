function escapeRegExp(value) {
    return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
export function extractModelDirective(body, options) {
    if (!body)
        return { cleaned: "", hasDirective: false };
    const modelMatch = body.match(/(?:^|\s)\/model(?=$|\s|:)\s*:?\s*([A-Za-z0-9_.:@-]+(?:\/[A-Za-z0-9_.:@-]+)*)?/i);
    const aliases = (options?.aliases ?? []).map((alias) => alias.trim()).filter(Boolean);
    const aliasMatch = modelMatch || aliases.length === 0
        ? null
        : body.match(new RegExp(`(?:^|\\s)\\/(${aliases.map(escapeRegExp).join("|")})(?=$|\\s|:)(?:\\s*:\\s*)?`, "i"));
    const match = modelMatch ?? aliasMatch;
    const raw = modelMatch ? modelMatch?.[1]?.trim() : aliasMatch?.[1]?.trim();
    let rawModel = raw;
    let rawProfile;
    if (raw?.includes("@")) {
        const parts = raw.split("@");
        rawModel = parts[0]?.trim();
        rawProfile = parts.slice(1).join("@").trim() || undefined;
    }
    const cleaned = match ? body.replace(match[0], " ").replace(/\s+/g, " ").trim() : body.trim();
    return {
        cleaned,
        rawModel,
        rawProfile,
        hasDirective: !!match,
    };
}
