const KEY_SPLIT_RE = /[\s,;]+/g;
function parseKeyList(raw) {
    if (!raw)
        return [];
    return raw
        .split(KEY_SPLIT_RE)
        .map((value) => value.trim())
        .filter(Boolean);
}
function collectEnvPrefixedKeys(prefix) {
    const keys = [];
    for (const [name, value] of Object.entries(process.env)) {
        if (!name.startsWith(prefix))
            continue;
        const trimmed = value?.trim();
        if (!trimmed)
            continue;
        keys.push(trimmed);
    }
    return keys;
}
export function collectAnthropicApiKeys() {
    const forcedSingle = process.env.CLAWDBOT_LIVE_ANTHROPIC_KEY?.trim();
    if (forcedSingle)
        return [forcedSingle];
    const fromList = parseKeyList(process.env.CLAWDBOT_LIVE_ANTHROPIC_KEYS);
    const fromEnv = collectEnvPrefixedKeys("ANTHROPIC_API_KEY");
    const primary = process.env.ANTHROPIC_API_KEY?.trim();
    const seen = new Set();
    const add = (value) => {
        if (!value)
            return;
        if (seen.has(value))
            return;
        seen.add(value);
    };
    for (const value of fromList)
        add(value);
    if (primary)
        add(primary);
    for (const value of fromEnv)
        add(value);
    return Array.from(seen);
}
export function isAnthropicRateLimitError(message) {
    const lower = message.toLowerCase();
    if (lower.includes("rate_limit"))
        return true;
    if (lower.includes("rate limit"))
        return true;
    if (lower.includes("429"))
        return true;
    return false;
}
export function isAnthropicBillingError(message) {
    const lower = message.toLowerCase();
    if (lower.includes("credit balance"))
        return true;
    if (lower.includes("insufficient credit"))
        return true;
    if (lower.includes("insufficient credits"))
        return true;
    if (lower.includes("payment required"))
        return true;
    if (lower.includes("billing") && lower.includes("disabled"))
        return true;
    if (lower.includes("402"))
        return true;
    return false;
}
