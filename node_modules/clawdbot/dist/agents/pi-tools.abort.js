function throwAbortError() {
    const err = new Error("Aborted");
    err.name = "AbortError";
    throw err;
}
function combineAbortSignals(a, b) {
    if (!a && !b)
        return undefined;
    if (a && !b)
        return a;
    if (b && !a)
        return b;
    if (a?.aborted)
        return a;
    if (b?.aborted)
        return b;
    if (typeof AbortSignal.any === "function") {
        return AbortSignal.any([a, b]);
    }
    const controller = new AbortController();
    const onAbort = () => controller.abort();
    a?.addEventListener("abort", onAbort, { once: true });
    b?.addEventListener("abort", onAbort, { once: true });
    return controller.signal;
}
export function wrapToolWithAbortSignal(tool, abortSignal) {
    if (!abortSignal)
        return tool;
    const execute = tool.execute;
    if (!execute)
        return tool;
    return {
        ...tool,
        execute: async (toolCallId, params, signal, onUpdate) => {
            const combined = combineAbortSignals(signal, abortSignal);
            if (combined?.aborted)
                throwAbortError();
            return await execute(toolCallId, params, combined, onUpdate);
        },
    };
}
