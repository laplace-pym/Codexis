import crypto from "node:crypto";
import net from "node:net";
import tls from "node:tls";
function normalizeFingerprint(input) {
    return input.replace(/[^a-fA-F0-9]/g, "").toLowerCase();
}
function extractFingerprint(raw) {
    const value = "fingerprint256" in raw ? raw.fingerprint256 : undefined;
    if (!value)
        return null;
    return normalizeFingerprint(value);
}
export class BridgeClient {
    opts;
    socket = null;
    buffer = "";
    pendingRpc = new Map();
    connected = false;
    helloReady = null;
    helloResolve = null;
    helloReject = null;
    constructor(opts) {
        this.opts = opts;
    }
    async connect() {
        if (this.connected)
            return;
        this.helloReady = new Promise((resolve, reject) => {
            this.helloResolve = resolve;
            this.helloReject = reject;
        });
        const socket = this.opts.tls
            ? tls.connect({
                host: this.opts.host,
                port: this.opts.port,
                rejectUnauthorized: false,
            })
            : net.connect({ host: this.opts.host, port: this.opts.port });
        this.socket = socket;
        socket.setNoDelay(true);
        socket.on("connect", () => {
            this.sendHello();
        });
        socket.on("error", (err) => {
            this.handleDisconnect(err);
        });
        socket.on("close", () => {
            this.handleDisconnect();
        });
        socket.on("data", (chunk) => {
            this.buffer += chunk.toString("utf8");
            this.flush();
        });
        if (this.opts.tls && socket instanceof tls.TLSSocket && this.opts.tlsFingerprint) {
            socket.once("secureConnect", () => {
                const cert = socket.getPeerCertificate(true);
                const fingerprint = cert ? extractFingerprint(cert) : null;
                if (!fingerprint || fingerprint !== normalizeFingerprint(this.opts.tlsFingerprint ?? "")) {
                    const err = new Error("bridge tls fingerprint mismatch");
                    this.handleDisconnect(err);
                    socket.destroy(err);
                }
            });
        }
        await this.helloReady;
    }
    async close() {
        if (this.socket) {
            this.socket.destroy();
            this.socket = null;
        }
        this.connected = false;
        this.pendingRpc.forEach((pending) => {
            if (pending.timer)
                clearTimeout(pending.timer);
            pending.reject(new Error("bridge client closed"));
        });
        this.pendingRpc.clear();
    }
    async request(method, params = null, timeoutMs = 5000) {
        const id = crypto.randomUUID();
        const frame = {
            type: "req",
            id,
            method,
            paramsJSON: params ? JSON.stringify(params) : null,
        };
        const res = await new Promise((resolve, reject) => {
            const timer = setTimeout(() => {
                this.pendingRpc.delete(id);
                reject(new Error(`bridge request timeout (${method})`));
            }, timeoutMs);
            this.pendingRpc.set(id, { resolve, reject, timer });
            this.send(frame);
        });
        if (!res.ok) {
            throw new Error(res.error?.message ?? "bridge request failed");
        }
        return res.payloadJSON ? JSON.parse(res.payloadJSON) : null;
    }
    sendEvent(event, payload) {
        const frame = {
            type: "event",
            event,
            payloadJSON: payload ? JSON.stringify(payload) : null,
        };
        this.send(frame);
    }
    sendInvokeResponse(frame) {
        this.send(frame);
    }
    sendHello() {
        const hello = {
            type: "hello",
            nodeId: this.opts.nodeId,
            token: this.opts.token,
            displayName: this.opts.displayName,
            platform: this.opts.platform,
            version: this.opts.version,
            coreVersion: this.opts.coreVersion,
            uiVersion: this.opts.uiVersion,
            deviceFamily: this.opts.deviceFamily,
            modelIdentifier: this.opts.modelIdentifier,
            caps: this.opts.caps,
            commands: this.opts.commands,
            permissions: this.opts.permissions,
        };
        this.send(hello);
    }
    sendPairRequest() {
        const req = {
            type: "pair-request",
            nodeId: this.opts.nodeId,
            displayName: this.opts.displayName,
            platform: this.opts.platform,
            version: this.opts.version,
            coreVersion: this.opts.coreVersion,
            uiVersion: this.opts.uiVersion,
            deviceFamily: this.opts.deviceFamily,
            modelIdentifier: this.opts.modelIdentifier,
            caps: this.opts.caps,
            commands: this.opts.commands,
            permissions: this.opts.permissions,
        };
        this.send(req);
    }
    send(frame) {
        if (!this.socket)
            return;
        this.socket.write(`${JSON.stringify(frame)}\n`);
    }
    handleDisconnect(err) {
        if (!this.connected && this.helloReject) {
            this.helloReject(err ?? new Error("bridge connection failed"));
            this.helloResolve = null;
            this.helloReject = null;
        }
        if (!this.connected && !this.socket)
            return;
        this.connected = false;
        this.socket = null;
        this.pendingRpc.forEach((pending) => {
            if (pending.timer)
                clearTimeout(pending.timer);
            pending.reject(err ?? new Error("bridge connection closed"));
        });
        this.pendingRpc.clear();
        void this.opts.onDisconnected?.(err);
    }
    flush() {
        while (true) {
            const idx = this.buffer.indexOf("\n");
            if (idx === -1)
                break;
            const line = this.buffer.slice(0, idx).trim();
            this.buffer = this.buffer.slice(idx + 1);
            if (!line)
                continue;
            let frame;
            try {
                frame = JSON.parse(line);
            }
            catch {
                continue;
            }
            this.handleFrame(frame);
        }
    }
    handleFrame(frame) {
        const type = String(frame.type ?? "");
        switch (type) {
            case "hello-ok": {
                this.connected = true;
                this.helloResolve?.();
                this.helloResolve = null;
                this.helloReject = null;
                void this.opts.onConnected?.(frame);
                return;
            }
            case "pair-ok": {
                const token = String(frame.token ?? "").trim();
                if (token) {
                    this.opts.token = token;
                    void this.opts.onPairToken?.(token);
                }
                return;
            }
            case "error": {
                const code = String(frame.code ?? "");
                if (code === "NOT_PAIRED" || code === "UNAUTHORIZED") {
                    this.opts.token = undefined;
                    void this.opts.onAuthReset?.();
                    this.sendPairRequest();
                    return;
                }
                this.handleDisconnect(new Error(frame.message ?? "bridge error"));
                return;
            }
            case "pong":
                return;
            case "ping": {
                const ping = frame;
                const pong = { type: "pong", id: String(ping.id ?? "") };
                this.send(pong);
                return;
            }
            case "event": {
                void this.opts.onEvent?.(frame);
                return;
            }
            case "res": {
                const res = frame;
                const pending = this.pendingRpc.get(res.id);
                if (pending) {
                    if (pending.timer)
                        clearTimeout(pending.timer);
                    this.pendingRpc.delete(res.id);
                    pending.resolve(res);
                }
                return;
            }
            case "invoke": {
                void this.opts.onInvoke?.(frame);
                return;
            }
            case "invoke-res": {
                return;
            }
            default:
                return;
        }
    }
}
