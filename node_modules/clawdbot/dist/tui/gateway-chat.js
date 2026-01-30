import { randomUUID } from "node:crypto";
import { loadConfig, resolveGatewayPort } from "../config/config.js";
import { GatewayClient } from "../gateway/client.js";
import { PROTOCOL_VERSION, } from "../gateway/protocol/index.js";
import { GATEWAY_CLIENT_MODES, GATEWAY_CLIENT_NAMES } from "../utils/message-channel.js";
import { VERSION } from "../version.js";
export class GatewayChatClient {
    client;
    readyPromise;
    resolveReady;
    connection;
    hello;
    onEvent;
    onConnected;
    onDisconnected;
    onGap;
    constructor(opts) {
        const resolved = resolveGatewayConnection(opts);
        this.connection = resolved;
        this.readyPromise = new Promise((resolve) => {
            this.resolveReady = resolve;
        });
        this.client = new GatewayClient({
            url: resolved.url,
            token: resolved.token,
            password: resolved.password,
            clientName: GATEWAY_CLIENT_NAMES.GATEWAY_CLIENT,
            clientDisplayName: "clawdbot-tui",
            clientVersion: VERSION,
            platform: process.platform,
            mode: GATEWAY_CLIENT_MODES.UI,
            instanceId: randomUUID(),
            minProtocol: PROTOCOL_VERSION,
            maxProtocol: PROTOCOL_VERSION,
            onHelloOk: (hello) => {
                this.hello = hello;
                this.resolveReady?.();
                this.onConnected?.();
            },
            onEvent: (evt) => {
                this.onEvent?.({
                    event: evt.event,
                    payload: evt.payload,
                    seq: evt.seq,
                });
            },
            onClose: (_code, reason) => {
                this.onDisconnected?.(reason);
            },
            onGap: (info) => {
                this.onGap?.(info);
            },
        });
    }
    start() {
        this.client.start();
    }
    stop() {
        this.client.stop();
    }
    async waitForReady() {
        await this.readyPromise;
    }
    async sendChat(opts) {
        const runId = randomUUID();
        await this.client.request("chat.send", {
            sessionKey: opts.sessionKey,
            message: opts.message,
            thinking: opts.thinking,
            deliver: opts.deliver,
            timeoutMs: opts.timeoutMs,
            idempotencyKey: runId,
        });
        return { runId };
    }
    async abortChat(opts) {
        return await this.client.request("chat.abort", {
            sessionKey: opts.sessionKey,
            runId: opts.runId,
        });
    }
    async loadHistory(opts) {
        return await this.client.request("chat.history", {
            sessionKey: opts.sessionKey,
            limit: opts.limit,
        });
    }
    async listSessions(opts) {
        return await this.client.request("sessions.list", {
            limit: opts?.limit,
            activeMinutes: opts?.activeMinutes,
            includeGlobal: opts?.includeGlobal,
            includeUnknown: opts?.includeUnknown,
            includeDerivedTitles: opts?.includeDerivedTitles,
            includeLastMessage: opts?.includeLastMessage,
            agentId: opts?.agentId,
        });
    }
    async listAgents() {
        return await this.client.request("agents.list", {});
    }
    async patchSession(opts) {
        return await this.client.request("sessions.patch", opts);
    }
    async resetSession(key) {
        return await this.client.request("sessions.reset", { key });
    }
    async getStatus() {
        return await this.client.request("status");
    }
    async listModels() {
        const res = await this.client.request("models.list");
        return Array.isArray(res?.models) ? res.models : [];
    }
}
export function resolveGatewayConnection(opts) {
    const config = loadConfig();
    const isRemoteMode = config.gateway?.mode === "remote";
    const remote = isRemoteMode ? config.gateway?.remote : undefined;
    const authToken = config.gateway?.auth?.token;
    const localPort = resolveGatewayPort(config);
    const url = (typeof opts.url === "string" && opts.url.trim().length > 0 ? opts.url.trim() : undefined) ||
        (typeof remote?.url === "string" && remote.url.trim().length > 0
            ? remote.url.trim()
            : undefined) ||
        `ws://127.0.0.1:${localPort}`;
    const token = (typeof opts.token === "string" && opts.token.trim().length > 0
        ? opts.token.trim()
        : undefined) ||
        (isRemoteMode
            ? typeof remote?.token === "string" && remote.token.trim().length > 0
                ? remote.token.trim()
                : undefined
            : process.env.CLAWDBOT_GATEWAY_TOKEN?.trim() ||
                (typeof authToken === "string" && authToken.trim().length > 0
                    ? authToken.trim()
                    : undefined));
    const password = (typeof opts.password === "string" && opts.password.trim().length > 0
        ? opts.password.trim()
        : undefined) ||
        process.env.CLAWDBOT_GATEWAY_PASSWORD?.trim() ||
        (typeof remote?.password === "string" && remote.password.trim().length > 0
            ? remote.password.trim()
            : undefined);
    return { url, token, password };
}
