export const normalizeAllowFrom = (list) => {
    const entries = (list ?? []).map((value) => String(value).trim()).filter(Boolean);
    const hasWildcard = entries.includes("*");
    const normalized = entries
        .filter((value) => value !== "*")
        .map((value) => value.replace(/^(telegram|tg):/i, ""));
    const normalizedLower = normalized.map((value) => value.toLowerCase());
    return {
        entries: normalized,
        entriesLower: normalizedLower,
        hasWildcard,
        hasEntries: entries.length > 0,
    };
};
export const normalizeAllowFromWithStore = (params) => {
    const combined = [...(params.allowFrom ?? []), ...(params.storeAllowFrom ?? [])]
        .map((value) => String(value).trim())
        .filter(Boolean);
    return normalizeAllowFrom(combined);
};
export const firstDefined = (...values) => {
    for (const value of values) {
        if (typeof value !== "undefined")
            return value;
    }
    return undefined;
};
export const isSenderAllowed = (params) => {
    const { allow, senderId, senderUsername } = params;
    if (!allow.hasEntries)
        return true;
    if (allow.hasWildcard)
        return true;
    if (senderId && allow.entries.includes(senderId))
        return true;
    const username = senderUsername?.toLowerCase();
    if (!username)
        return false;
    return allow.entriesLower.some((entry) => entry === username || entry === `@${username}`);
};
export const resolveSenderAllowMatch = (params) => {
    const { allow, senderId, senderUsername } = params;
    if (allow.hasWildcard) {
        return { allowed: true, matchKey: "*", matchSource: "wildcard" };
    }
    if (!allow.hasEntries)
        return { allowed: false };
    if (senderId && allow.entries.includes(senderId)) {
        return { allowed: true, matchKey: senderId, matchSource: "id" };
    }
    const username = senderUsername?.toLowerCase();
    if (!username)
        return { allowed: false };
    const entry = allow.entriesLower.find((candidate) => candidate === username || candidate === `@${username}`);
    if (entry) {
        return { allowed: true, matchKey: entry, matchSource: "username" };
    }
    return { allowed: false };
};
