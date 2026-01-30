const THREAD_SUFFIX_REGEX = /^(.*)(?::(?:thread|topic):\d+)$/i;
function stripThreadSuffix(value) {
    const match = value.match(THREAD_SUFFIX_REGEX);
    return match?.[1] ?? value;
}
/**
 * Limits conversation history to the last N user turns (and their associated
 * assistant responses). This reduces token usage for long-running DM sessions.
 */
export function limitHistoryTurns(messages, limit) {
    if (!limit || limit <= 0 || messages.length === 0)
        return messages;
    let userCount = 0;
    let lastUserIndex = messages.length;
    for (let i = messages.length - 1; i >= 0; i--) {
        if (messages[i].role === "user") {
            userCount++;
            if (userCount > limit) {
                return messages.slice(lastUserIndex);
            }
            lastUserIndex = i;
        }
    }
    return messages;
}
/**
 * Extract provider + user ID from a session key and look up dmHistoryLimit.
 * Supports per-DM overrides and provider defaults.
 */
export function getDmHistoryLimitFromSessionKey(sessionKey, config) {
    if (!sessionKey || !config)
        return undefined;
    const parts = sessionKey.split(":").filter(Boolean);
    const providerParts = parts.length >= 3 && parts[0] === "agent" ? parts.slice(2) : parts;
    const provider = providerParts[0]?.toLowerCase();
    if (!provider)
        return undefined;
    const kind = providerParts[1]?.toLowerCase();
    const userIdRaw = providerParts.slice(2).join(":");
    const userId = stripThreadSuffix(userIdRaw);
    if (kind !== "dm")
        return undefined;
    const getLimit = (providerConfig) => {
        if (!providerConfig)
            return undefined;
        if (userId && providerConfig.dms?.[userId]?.historyLimit !== undefined) {
            return providerConfig.dms[userId].historyLimit;
        }
        return providerConfig.dmHistoryLimit;
    };
    const resolveProviderConfig = (cfg, providerId) => {
        const channels = cfg?.channels;
        if (!channels || typeof channels !== "object")
            return undefined;
        const entry = channels[providerId];
        if (!entry || typeof entry !== "object" || Array.isArray(entry))
            return undefined;
        return entry;
    };
    return getLimit(resolveProviderConfig(config, provider));
}
