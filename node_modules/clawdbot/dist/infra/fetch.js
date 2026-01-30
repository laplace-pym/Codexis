function withDuplex(init, input) {
    const hasInitBody = init?.body != null;
    const hasRequestBody = !hasInitBody &&
        typeof Request !== "undefined" &&
        input instanceof Request &&
        input.body != null;
    if (!hasInitBody && !hasRequestBody)
        return init;
    if (init && "duplex" in init)
        return init;
    return init
        ? { ...init, duplex: "half" }
        : { duplex: "half" };
}
export function wrapFetchWithAbortSignal(fetchImpl) {
    const wrapped = ((input, init) => {
        const patchedInit = withDuplex(init, input);
        const signal = patchedInit?.signal;
        if (!signal)
            return fetchImpl(input, patchedInit);
        if (typeof AbortSignal !== "undefined" && signal instanceof AbortSignal) {
            return fetchImpl(input, patchedInit);
        }
        if (typeof AbortController === "undefined") {
            return fetchImpl(input, patchedInit);
        }
        if (typeof signal.addEventListener !== "function") {
            return fetchImpl(input, patchedInit);
        }
        const controller = new AbortController();
        const onAbort = () => controller.abort();
        if (signal.aborted) {
            controller.abort();
        }
        else {
            signal.addEventListener("abort", onAbort, { once: true });
        }
        const response = fetchImpl(input, { ...patchedInit, signal: controller.signal });
        if (typeof signal.removeEventListener === "function") {
            void response.finally(() => {
                signal.removeEventListener("abort", onAbort);
            });
        }
        return response;
    });
    const fetchWithPreconnect = fetchImpl;
    wrapped.preconnect =
        typeof fetchWithPreconnect.preconnect === "function"
            ? fetchWithPreconnect.preconnect.bind(fetchWithPreconnect)
            : () => { };
    return Object.assign(wrapped, fetchImpl);
}
export function resolveFetch(fetchImpl) {
    const resolved = fetchImpl ?? globalThis.fetch;
    if (!resolved)
        return undefined;
    return wrapFetchWithAbortSignal(resolved);
}
