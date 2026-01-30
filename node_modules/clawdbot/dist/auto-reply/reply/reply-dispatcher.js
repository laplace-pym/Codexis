import { normalizeReplyPayload } from "./normalize-reply.js";
const DEFAULT_HUMAN_DELAY_MIN_MS = 800;
const DEFAULT_HUMAN_DELAY_MAX_MS = 2500;
/** Generate a random delay within the configured range. */
function getHumanDelay(config) {
    const mode = config?.mode ?? "off";
    if (mode === "off")
        return 0;
    const min = mode === "custom" ? (config?.minMs ?? DEFAULT_HUMAN_DELAY_MIN_MS) : DEFAULT_HUMAN_DELAY_MIN_MS;
    const max = mode === "custom" ? (config?.maxMs ?? DEFAULT_HUMAN_DELAY_MAX_MS) : DEFAULT_HUMAN_DELAY_MAX_MS;
    if (max <= min)
        return min;
    return Math.floor(Math.random() * (max - min + 1)) + min;
}
/** Sleep for a given number of milliseconds. */
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
function normalizeReplyPayloadInternal(payload, opts) {
    // Prefer dynamic context provider over static context
    const prefixContext = opts.responsePrefixContextProvider?.() ?? opts.responsePrefixContext;
    return normalizeReplyPayload(payload, {
        responsePrefix: opts.responsePrefix,
        responsePrefixContext: prefixContext,
        onHeartbeatStrip: opts.onHeartbeatStrip,
    });
}
export function createReplyDispatcher(options) {
    let sendChain = Promise.resolve();
    // Track in-flight deliveries so we can emit a reliable "idle" signal.
    let pending = 0;
    // Track whether we've sent a block reply (for human delay - skip delay on first block).
    let sentFirstBlock = false;
    // Serialize outbound replies to preserve tool/block/final order.
    const queuedCounts = {
        tool: 0,
        block: 0,
        final: 0,
    };
    const enqueue = (kind, payload) => {
        const normalized = normalizeReplyPayloadInternal(payload, options);
        if (!normalized)
            return false;
        queuedCounts[kind] += 1;
        pending += 1;
        // Determine if we should add human-like delay (only for block replies after the first).
        const shouldDelay = kind === "block" && sentFirstBlock;
        if (kind === "block")
            sentFirstBlock = true;
        sendChain = sendChain
            .then(async () => {
            // Add human-like delay between block replies for natural rhythm.
            if (shouldDelay) {
                const delayMs = getHumanDelay(options.humanDelay);
                if (delayMs > 0)
                    await sleep(delayMs);
            }
            await options.deliver(normalized, { kind });
        })
            .catch((err) => {
            options.onError?.(err, { kind });
        })
            .finally(() => {
            pending -= 1;
            if (pending === 0) {
                options.onIdle?.();
            }
        });
        return true;
    };
    return {
        sendToolResult: (payload) => enqueue("tool", payload),
        sendBlockReply: (payload) => enqueue("block", payload),
        sendFinalReply: (payload) => enqueue("final", payload),
        waitForIdle: () => sendChain,
        getQueuedCounts: () => ({ ...queuedCounts }),
    };
}
export function createReplyDispatcherWithTyping(options) {
    const { onReplyStart, onIdle, ...dispatcherOptions } = options;
    let typingController;
    const dispatcher = createReplyDispatcher({
        ...dispatcherOptions,
        onIdle: () => {
            typingController?.markDispatchIdle();
            onIdle?.();
        },
    });
    return {
        dispatcher,
        replyOptions: {
            onReplyStart,
            onTypingController: (typing) => {
                typingController = typing;
            },
        },
        markDispatchIdle: () => {
            typingController?.markDispatchIdle();
            onIdle?.();
        },
    };
}
