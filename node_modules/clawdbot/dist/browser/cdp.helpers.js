import WebSocket from "ws";
import { rawDataToString } from "../infra/ws.js";
export function isLoopbackHost(host) {
    const h = host.trim().toLowerCase();
    return (h === "localhost" ||
        h === "127.0.0.1" ||
        h === "0.0.0.0" ||
        h === "[::1]" ||
        h === "::1" ||
        h === "[::]" ||
        h === "::");
}
export function getHeadersWithAuth(url, headers = {}) {
    try {
        const parsed = new URL(url);
        const hasAuthHeader = Object.keys(headers).some((key) => key.toLowerCase() === "authorization");
        if (hasAuthHeader)
            return headers;
        if (parsed.username || parsed.password) {
            const auth = Buffer.from(`${parsed.username}:${parsed.password}`).toString("base64");
            return { ...headers, Authorization: `Basic ${auth}` };
        }
    }
    catch {
        // ignore
    }
    return headers;
}
export function appendCdpPath(cdpUrl, path) {
    const url = new URL(cdpUrl);
    const basePath = url.pathname.replace(/\/$/, "");
    const suffix = path.startsWith("/") ? path : `/${path}`;
    url.pathname = `${basePath}${suffix}`;
    return url.toString();
}
function createCdpSender(ws) {
    let nextId = 1;
    const pending = new Map();
    const send = (method, params) => {
        const id = nextId++;
        const msg = { id, method, params };
        ws.send(JSON.stringify(msg));
        return new Promise((resolve, reject) => {
            pending.set(id, { resolve, reject });
        });
    };
    const closeWithError = (err) => {
        for (const [, p] of pending)
            p.reject(err);
        pending.clear();
        try {
            ws.close();
        }
        catch {
            // ignore
        }
    };
    ws.on("message", (data) => {
        try {
            const parsed = JSON.parse(rawDataToString(data));
            if (typeof parsed.id !== "number")
                return;
            const p = pending.get(parsed.id);
            if (!p)
                return;
            pending.delete(parsed.id);
            if (parsed.error?.message) {
                p.reject(new Error(parsed.error.message));
                return;
            }
            p.resolve(parsed.result);
        }
        catch {
            // ignore
        }
    });
    ws.on("close", () => {
        closeWithError(new Error("CDP socket closed"));
    });
    return { send, closeWithError };
}
export async function fetchJson(url, timeoutMs = 1500, init) {
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), timeoutMs);
    try {
        const headers = getHeadersWithAuth(url, init?.headers || {});
        const res = await fetch(url, { ...init, headers, signal: ctrl.signal });
        if (!res.ok)
            throw new Error(`HTTP ${res.status}`);
        return (await res.json());
    }
    finally {
        clearTimeout(t);
    }
}
export async function fetchOk(url, timeoutMs = 1500, init) {
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), timeoutMs);
    try {
        const headers = getHeadersWithAuth(url, init?.headers || {});
        const res = await fetch(url, { ...init, headers, signal: ctrl.signal });
        if (!res.ok)
            throw new Error(`HTTP ${res.status}`);
    }
    finally {
        clearTimeout(t);
    }
}
export async function withCdpSocket(wsUrl, fn, opts) {
    const headers = getHeadersWithAuth(wsUrl, opts?.headers ?? {});
    const ws = new WebSocket(wsUrl, {
        handshakeTimeout: 5000,
        ...(Object.keys(headers).length ? { headers } : {}),
    });
    const { send, closeWithError } = createCdpSender(ws);
    const openPromise = new Promise((resolve, reject) => {
        ws.once("open", () => resolve());
        ws.once("error", (err) => reject(err));
    });
    await openPromise;
    try {
        return await fn(send);
    }
    catch (err) {
        closeWithError(err instanceof Error ? err : new Error(String(err)));
        throw err;
    }
    finally {
        try {
            ws.close();
        }
        catch {
            // ignore
        }
    }
}
