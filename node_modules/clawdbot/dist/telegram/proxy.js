// @ts-nocheck
import { ProxyAgent } from "undici";
import { wrapFetchWithAbortSignal } from "../infra/fetch.js";
export function makeProxyFetch(proxyUrl) {
    const agent = new ProxyAgent(proxyUrl);
    return wrapFetchWithAbortSignal((input, init) => {
        const base = init ? { ...init } : {};
        return fetch(input, { ...base, dispatcher: agent });
    });
}
