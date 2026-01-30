const resolveMs = (value) => {
    if (typeof value !== "number" || !Number.isFinite(value))
        return undefined;
    return Math.max(0, Math.trunc(value));
};
const resolveChannelOverride = (params) => {
    if (!params.byChannel)
        return undefined;
    return resolveMs(params.byChannel[params.channel]);
};
export function resolveInboundDebounceMs(params) {
    const inbound = params.cfg.messages?.inbound;
    const override = resolveMs(params.overrideMs);
    const byChannel = resolveChannelOverride({
        byChannel: inbound?.byChannel,
        channel: params.channel,
    });
    const base = resolveMs(inbound?.debounceMs);
    return override ?? byChannel ?? base ?? 0;
}
export function createInboundDebouncer(params) {
    const buffers = new Map();
    const debounceMs = Math.max(0, Math.trunc(params.debounceMs));
    const flushBuffer = async (key, buffer) => {
        buffers.delete(key);
        if (buffer.timeout) {
            clearTimeout(buffer.timeout);
            buffer.timeout = null;
        }
        if (buffer.items.length === 0)
            return;
        try {
            await params.onFlush(buffer.items);
        }
        catch (err) {
            params.onError?.(err, buffer.items);
        }
    };
    const flushKey = async (key) => {
        const buffer = buffers.get(key);
        if (!buffer)
            return;
        await flushBuffer(key, buffer);
    };
    const scheduleFlush = (key, buffer) => {
        if (buffer.timeout)
            clearTimeout(buffer.timeout);
        buffer.timeout = setTimeout(() => {
            void flushBuffer(key, buffer);
        }, debounceMs);
        buffer.timeout.unref?.();
    };
    const enqueue = async (item) => {
        const key = params.buildKey(item);
        const canDebounce = debounceMs > 0 && (params.shouldDebounce?.(item) ?? true);
        if (!canDebounce || !key) {
            if (key && buffers.has(key)) {
                await flushKey(key);
            }
            await params.onFlush([item]);
            return;
        }
        const existing = buffers.get(key);
        if (existing) {
            existing.items.push(item);
            scheduleFlush(key, existing);
            return;
        }
        const buffer = { items: [item], timeout: null };
        buffers.set(key, buffer);
        scheduleFlush(key, buffer);
    };
    return { enqueue, flushKey };
}
