export const DEFAULT_TIMEOUT_SECONDS = 30;
export const DEFAULT_CACHE_TTL_MINUTES = 15;
const DEFAULT_CACHE_MAX_ENTRIES = 100;
export function resolveTimeoutSeconds(value, fallback) {
    const parsed = typeof value === "number" && Number.isFinite(value) ? value : fallback;
    return Math.max(1, Math.floor(parsed));
}
export function resolveCacheTtlMs(value, fallbackMinutes) {
    const minutes = typeof value === "number" && Number.isFinite(value) ? Math.max(0, value) : fallbackMinutes;
    return Math.round(minutes * 60_000);
}
export function normalizeCacheKey(value) {
    return value.trim().toLowerCase();
}
export function readCache(cache, key) {
    const entry = cache.get(key);
    if (!entry)
        return null;
    if (Date.now() > entry.expiresAt) {
        cache.delete(key);
        return null;
    }
    return { value: entry.value, cached: true };
}
export function writeCache(cache, key, value, ttlMs) {
    if (ttlMs <= 0)
        return;
    if (cache.size >= DEFAULT_CACHE_MAX_ENTRIES) {
        const oldest = cache.keys().next();
        if (!oldest.done)
            cache.delete(oldest.value);
    }
    cache.set(key, {
        value,
        expiresAt: Date.now() + ttlMs,
        insertedAt: Date.now(),
    });
}
export function withTimeout(signal, timeoutMs) {
    if (timeoutMs <= 0)
        return signal ?? new AbortController().signal;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    if (signal) {
        signal.addEventListener("abort", () => {
            clearTimeout(timer);
            controller.abort();
        }, { once: true });
    }
    controller.signal.addEventListener("abort", () => {
        clearTimeout(timer);
    }, { once: true });
    return controller.signal;
}
export async function readResponseText(res) {
    try {
        return await res.text();
    }
    catch {
        return "";
    }
}
