import { randomUUID } from "node:crypto";
import { PROTOCOL_VERSION } from "@agentclientprotocol/sdk";
import { getAvailableCommands } from "./commands.js";
import { readBool, readNumber, readString } from "./meta.js";
import { extractAttachmentsFromPrompt, extractTextFromPrompt, formatToolTitle, inferToolKind, } from "./event-mapper.js";
import { parseSessionMeta, resetSessionIfNeeded, resolveSessionKey } from "./session-mapper.js";
import { ACP_AGENT_INFO } from "./types.js";
import { defaultAcpSessionStore } from "./session.js";
export class AcpGatewayAgent {
    connection;
    gateway;
    opts;
    log;
    sessionStore;
    pendingPrompts = new Map();
    constructor(connection, gateway, opts = {}) {
        this.connection = connection;
        this.gateway = gateway;
        this.opts = opts;
        this.log = opts.verbose ? (msg) => process.stderr.write(`[acp] ${msg}\n`) : () => { };
        this.sessionStore = opts.sessionStore ?? defaultAcpSessionStore;
    }
    start() {
        this.log("ready");
    }
    handleGatewayReconnect() {
        this.log("gateway reconnected");
    }
    handleGatewayDisconnect(reason) {
        this.log(`gateway disconnected: ${reason}`);
        for (const pending of this.pendingPrompts.values()) {
            pending.reject(new Error(`Gateway disconnected: ${reason}`));
            this.sessionStore.clearActiveRun(pending.sessionId);
        }
        this.pendingPrompts.clear();
    }
    async handleGatewayEvent(evt) {
        if (evt.event === "chat") {
            await this.handleChatEvent(evt);
            return;
        }
        if (evt.event === "agent") {
            await this.handleAgentEvent(evt);
        }
    }
    async initialize(_params) {
        return {
            protocolVersion: PROTOCOL_VERSION,
            agentCapabilities: {
                loadSession: true,
                promptCapabilities: {
                    image: true,
                    audio: false,
                    embeddedContext: true,
                },
                mcpCapabilities: {
                    http: false,
                    sse: false,
                },
                sessionCapabilities: {
                    list: {},
                },
            },
            agentInfo: ACP_AGENT_INFO,
            authMethods: [],
        };
    }
    async newSession(params) {
        if (params.mcpServers.length > 0) {
            this.log(`ignoring ${params.mcpServers.length} MCP servers`);
        }
        const sessionId = randomUUID();
        const meta = parseSessionMeta(params._meta);
        const sessionKey = await resolveSessionKey({
            meta,
            fallbackKey: `acp:${sessionId}`,
            gateway: this.gateway,
            opts: this.opts,
        });
        await resetSessionIfNeeded({
            meta,
            sessionKey,
            gateway: this.gateway,
            opts: this.opts,
        });
        const session = this.sessionStore.createSession({
            sessionId,
            sessionKey,
            cwd: params.cwd,
        });
        this.log(`newSession: ${session.sessionId} -> ${session.sessionKey}`);
        await this.sendAvailableCommands(session.sessionId);
        return { sessionId: session.sessionId };
    }
    async loadSession(params) {
        if (params.mcpServers.length > 0) {
            this.log(`ignoring ${params.mcpServers.length} MCP servers`);
        }
        const meta = parseSessionMeta(params._meta);
        const sessionKey = await resolveSessionKey({
            meta,
            fallbackKey: params.sessionId,
            gateway: this.gateway,
            opts: this.opts,
        });
        await resetSessionIfNeeded({
            meta,
            sessionKey,
            gateway: this.gateway,
            opts: this.opts,
        });
        const session = this.sessionStore.createSession({
            sessionId: params.sessionId,
            sessionKey,
            cwd: params.cwd,
        });
        this.log(`loadSession: ${session.sessionId} -> ${session.sessionKey}`);
        await this.sendAvailableCommands(session.sessionId);
        return {};
    }
    async unstable_listSessions(params) {
        const limit = readNumber(params._meta, ["limit"]) ?? 100;
        const result = await this.gateway.request("sessions.list", { limit });
        const cwd = params.cwd ?? process.cwd();
        return {
            sessions: result.sessions.map((session) => ({
                sessionId: session.key,
                cwd,
                title: session.displayName ?? session.label ?? session.key,
                updatedAt: session.updatedAt ? new Date(session.updatedAt).toISOString() : undefined,
                _meta: {
                    sessionKey: session.key,
                    kind: session.kind,
                    channel: session.channel,
                },
            })),
            nextCursor: null,
        };
    }
    async authenticate(_params) {
        return {};
    }
    async setSessionMode(params) {
        const session = this.sessionStore.getSession(params.sessionId);
        if (!session) {
            throw new Error(`Session ${params.sessionId} not found`);
        }
        if (!params.modeId)
            return {};
        try {
            await this.gateway.request("sessions.patch", {
                key: session.sessionKey,
                thinkingLevel: params.modeId,
            });
            this.log(`setSessionMode: ${session.sessionId} -> ${params.modeId}`);
        }
        catch (err) {
            this.log(`setSessionMode error: ${String(err)}`);
        }
        return {};
    }
    async prompt(params) {
        const session = this.sessionStore.getSession(params.sessionId);
        if (!session) {
            throw new Error(`Session ${params.sessionId} not found`);
        }
        if (session.abortController) {
            this.sessionStore.cancelActiveRun(params.sessionId);
        }
        const abortController = new AbortController();
        const runId = randomUUID();
        this.sessionStore.setActiveRun(params.sessionId, runId, abortController);
        const meta = parseSessionMeta(params._meta);
        const userText = extractTextFromPrompt(params.prompt);
        const attachments = extractAttachmentsFromPrompt(params.prompt);
        const prefixCwd = meta.prefixCwd ?? this.opts.prefixCwd ?? true;
        const message = prefixCwd ? `[Working directory: ${session.cwd}]\n\n${userText}` : userText;
        return new Promise((resolve, reject) => {
            this.pendingPrompts.set(params.sessionId, {
                sessionId: params.sessionId,
                sessionKey: session.sessionKey,
                idempotencyKey: runId,
                resolve,
                reject,
            });
            this.gateway
                .request("chat.send", {
                sessionKey: session.sessionKey,
                message,
                attachments: attachments.length > 0 ? attachments : undefined,
                idempotencyKey: runId,
                thinking: readString(params._meta, ["thinking", "thinkingLevel"]),
                deliver: readBool(params._meta, ["deliver"]),
                timeoutMs: readNumber(params._meta, ["timeoutMs"]),
            }, { expectFinal: true })
                .catch((err) => {
                this.pendingPrompts.delete(params.sessionId);
                this.sessionStore.clearActiveRun(params.sessionId);
                reject(err instanceof Error ? err : new Error(String(err)));
            });
        });
    }
    async cancel(params) {
        const session = this.sessionStore.getSession(params.sessionId);
        if (!session)
            return;
        this.sessionStore.cancelActiveRun(params.sessionId);
        try {
            await this.gateway.request("chat.abort", { sessionKey: session.sessionKey });
        }
        catch (err) {
            this.log(`cancel error: ${String(err)}`);
        }
        const pending = this.pendingPrompts.get(params.sessionId);
        if (pending) {
            this.pendingPrompts.delete(params.sessionId);
            pending.resolve({ stopReason: "cancelled" });
        }
    }
    async handleAgentEvent(evt) {
        const payload = evt.payload;
        if (!payload)
            return;
        const stream = payload.stream;
        const data = payload.data;
        const sessionKey = payload.sessionKey;
        if (!stream || !data || !sessionKey)
            return;
        if (stream !== "tool")
            return;
        const phase = data.phase;
        const name = data.name;
        const toolCallId = data.toolCallId;
        if (!toolCallId)
            return;
        const pending = this.findPendingBySessionKey(sessionKey);
        if (!pending)
            return;
        if (phase === "start") {
            if (!pending.toolCalls)
                pending.toolCalls = new Set();
            if (pending.toolCalls.has(toolCallId))
                return;
            pending.toolCalls.add(toolCallId);
            const args = data.args;
            await this.connection.sessionUpdate({
                sessionId: pending.sessionId,
                update: {
                    sessionUpdate: "tool_call",
                    toolCallId,
                    title: formatToolTitle(name, args),
                    status: "in_progress",
                    rawInput: args,
                    kind: inferToolKind(name),
                },
            });
            return;
        }
        if (phase === "result") {
            const isError = Boolean(data.isError);
            await this.connection.sessionUpdate({
                sessionId: pending.sessionId,
                update: {
                    sessionUpdate: "tool_call_update",
                    toolCallId,
                    status: isError ? "failed" : "completed",
                    rawOutput: data.result,
                },
            });
        }
    }
    async handleChatEvent(evt) {
        const payload = evt.payload;
        if (!payload)
            return;
        const sessionKey = payload.sessionKey;
        const state = payload.state;
        const runId = payload.runId;
        const messageData = payload.message;
        if (!sessionKey || !state)
            return;
        const pending = this.findPendingBySessionKey(sessionKey);
        if (!pending)
            return;
        if (runId && pending.idempotencyKey !== runId)
            return;
        if (state === "delta" && messageData) {
            await this.handleDeltaEvent(pending.sessionId, messageData);
            return;
        }
        if (state === "final") {
            this.finishPrompt(pending.sessionId, pending, "end_turn");
            return;
        }
        if (state === "aborted") {
            this.finishPrompt(pending.sessionId, pending, "cancelled");
            return;
        }
        if (state === "error") {
            this.finishPrompt(pending.sessionId, pending, "refusal");
        }
    }
    async handleDeltaEvent(sessionId, messageData) {
        const content = messageData.content;
        const fullText = content?.find((c) => c.type === "text")?.text ?? "";
        const pending = this.pendingPrompts.get(sessionId);
        if (!pending)
            return;
        const sentSoFar = pending.sentTextLength ?? 0;
        if (fullText.length <= sentSoFar)
            return;
        const newText = fullText.slice(sentSoFar);
        pending.sentTextLength = fullText.length;
        pending.sentText = fullText;
        await this.connection.sessionUpdate({
            sessionId,
            update: {
                sessionUpdate: "agent_message_chunk",
                content: { type: "text", text: newText },
            },
        });
    }
    finishPrompt(sessionId, pending, stopReason) {
        this.pendingPrompts.delete(sessionId);
        this.sessionStore.clearActiveRun(sessionId);
        pending.resolve({ stopReason });
    }
    findPendingBySessionKey(sessionKey) {
        for (const pending of this.pendingPrompts.values()) {
            if (pending.sessionKey === sessionKey)
                return pending;
        }
        return undefined;
    }
    async sendAvailableCommands(sessionId) {
        await this.connection.sessionUpdate({
            sessionId,
            update: {
                sessionUpdate: "available_commands_update",
                availableCommands: getAvailableCommands(),
            },
        });
    }
}
