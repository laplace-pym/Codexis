const DEFAULT_AGENT_TIMEOUT_SECONDS = 600;
const normalizeNumber = (value) => typeof value === "number" && Number.isFinite(value) ? Math.floor(value) : undefined;
export function resolveAgentTimeoutSeconds(cfg) {
    const raw = normalizeNumber(cfg?.agents?.defaults?.timeoutSeconds);
    const seconds = raw ?? DEFAULT_AGENT_TIMEOUT_SECONDS;
    return Math.max(seconds, 1);
}
export function resolveAgentTimeoutMs(opts) {
    const minMs = Math.max(normalizeNumber(opts.minMs) ?? 1, 1);
    const defaultMs = resolveAgentTimeoutSeconds(opts.cfg) * 1000;
    const overrideMs = normalizeNumber(opts.overrideMs);
    if (overrideMs !== undefined) {
        if (overrideMs <= 0)
            return defaultMs;
        return Math.max(overrideMs, minMs);
    }
    const overrideSeconds = normalizeNumber(opts.overrideSeconds);
    if (overrideSeconds !== undefined) {
        if (overrideSeconds <= 0)
            return defaultMs;
        return Math.max(overrideSeconds * 1000, minMs);
    }
    return Math.max(defaultMs, minMs);
}
