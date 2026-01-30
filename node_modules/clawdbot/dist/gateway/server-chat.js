import { normalizeVerboseLevel } from "../auto-reply/thinking.js";
import { getAgentRunContext } from "../infra/agent-events.js";
import { loadSessionEntry } from "./session-utils.js";
import { formatForLog } from "./ws-log.js";
export function createChatRunRegistry() {
    const chatRunSessions = new Map();
    const add = (sessionId, entry) => {
        const queue = chatRunSessions.get(sessionId);
        if (queue) {
            queue.push(entry);
        }
        else {
            chatRunSessions.set(sessionId, [entry]);
        }
    };
    const peek = (sessionId) => chatRunSessions.get(sessionId)?.[0];
    const shift = (sessionId) => {
        const queue = chatRunSessions.get(sessionId);
        if (!queue || queue.length === 0)
            return undefined;
        const entry = queue.shift();
        if (!queue.length)
            chatRunSessions.delete(sessionId);
        return entry;
    };
    const remove = (sessionId, clientRunId, sessionKey) => {
        const queue = chatRunSessions.get(sessionId);
        if (!queue || queue.length === 0)
            return undefined;
        const idx = queue.findIndex((entry) => entry.clientRunId === clientRunId && (sessionKey ? entry.sessionKey === sessionKey : true));
        if (idx < 0)
            return undefined;
        const [entry] = queue.splice(idx, 1);
        if (!queue.length)
            chatRunSessions.delete(sessionId);
        return entry;
    };
    const clear = () => {
        chatRunSessions.clear();
    };
    return { add, peek, shift, remove, clear };
}
export function createChatRunState() {
    const registry = createChatRunRegistry();
    const buffers = new Map();
    const deltaSentAt = new Map();
    const abortedRuns = new Map();
    const clear = () => {
        registry.clear();
        buffers.clear();
        deltaSentAt.clear();
        abortedRuns.clear();
    };
    return {
        registry,
        buffers,
        deltaSentAt,
        abortedRuns,
        clear,
    };
}
export function createAgentEventHandler({ broadcast, nodeSendToSession, agentRunSeq, chatRunState, resolveSessionKeyForRun, clearAgentRunContext, }) {
    const emitChatDelta = (sessionKey, clientRunId, seq, text) => {
        chatRunState.buffers.set(clientRunId, text);
        const now = Date.now();
        const last = chatRunState.deltaSentAt.get(clientRunId) ?? 0;
        if (now - last < 150)
            return;
        chatRunState.deltaSentAt.set(clientRunId, now);
        const payload = {
            runId: clientRunId,
            sessionKey,
            seq,
            state: "delta",
            message: {
                role: "assistant",
                content: [{ type: "text", text }],
                timestamp: now,
            },
        };
        broadcast("chat", payload, { dropIfSlow: true });
        nodeSendToSession(sessionKey, "chat", payload);
    };
    const emitChatFinal = (sessionKey, clientRunId, seq, jobState, error) => {
        const text = chatRunState.buffers.get(clientRunId)?.trim() ?? "";
        chatRunState.buffers.delete(clientRunId);
        chatRunState.deltaSentAt.delete(clientRunId);
        if (jobState === "done") {
            const payload = {
                runId: clientRunId,
                sessionKey,
                seq,
                state: "final",
                message: text
                    ? {
                        role: "assistant",
                        content: [{ type: "text", text }],
                        timestamp: Date.now(),
                    }
                    : undefined,
            };
            broadcast("chat", payload);
            nodeSendToSession(sessionKey, "chat", payload);
            return;
        }
        const payload = {
            runId: clientRunId,
            sessionKey,
            seq,
            state: "error",
            errorMessage: error ? formatForLog(error) : undefined,
        };
        broadcast("chat", payload);
        nodeSendToSession(sessionKey, "chat", payload);
    };
    const shouldEmitToolEvents = (runId, sessionKey) => {
        const runContext = getAgentRunContext(runId);
        const runVerbose = normalizeVerboseLevel(runContext?.verboseLevel);
        if (runVerbose)
            return runVerbose === "on";
        if (!sessionKey)
            return false;
        try {
            const { cfg, entry } = loadSessionEntry(sessionKey);
            const sessionVerbose = normalizeVerboseLevel(entry?.verboseLevel);
            if (sessionVerbose)
                return sessionVerbose === "on";
            const defaultVerbose = normalizeVerboseLevel(cfg.agents?.defaults?.verboseDefault);
            return defaultVerbose === "on";
        }
        catch {
            return false;
        }
    };
    return (evt) => {
        const chatLink = chatRunState.registry.peek(evt.runId);
        const sessionKey = chatLink?.sessionKey ?? resolveSessionKeyForRun(evt.runId);
        const clientRunId = chatLink?.clientRunId ?? evt.runId;
        const isAborted = chatRunState.abortedRuns.has(clientRunId) || chatRunState.abortedRuns.has(evt.runId);
        // Include sessionKey so Control UI can filter tool streams per session.
        const agentPayload = sessionKey ? { ...evt, sessionKey } : evt;
        const last = agentRunSeq.get(evt.runId) ?? 0;
        if (evt.stream === "tool" && !shouldEmitToolEvents(evt.runId, sessionKey)) {
            agentRunSeq.set(evt.runId, evt.seq);
            return;
        }
        if (evt.seq !== last + 1) {
            broadcast("agent", {
                runId: evt.runId,
                stream: "error",
                ts: Date.now(),
                sessionKey,
                data: {
                    reason: "seq gap",
                    expected: last + 1,
                    received: evt.seq,
                },
            });
        }
        agentRunSeq.set(evt.runId, evt.seq);
        broadcast("agent", agentPayload);
        const lifecyclePhase = evt.stream === "lifecycle" && typeof evt.data?.phase === "string" ? evt.data.phase : null;
        if (sessionKey) {
            nodeSendToSession(sessionKey, "agent", agentPayload);
            if (!isAborted && evt.stream === "assistant" && typeof evt.data?.text === "string") {
                emitChatDelta(sessionKey, clientRunId, evt.seq, evt.data.text);
            }
            else if (!isAborted && (lifecyclePhase === "end" || lifecyclePhase === "error")) {
                if (chatLink) {
                    const finished = chatRunState.registry.shift(evt.runId);
                    if (!finished) {
                        clearAgentRunContext(evt.runId);
                        return;
                    }
                    emitChatFinal(finished.sessionKey, finished.clientRunId, evt.seq, lifecyclePhase === "error" ? "error" : "done", evt.data?.error);
                }
                else {
                    emitChatFinal(sessionKey, evt.runId, evt.seq, lifecyclePhase === "error" ? "error" : "done", evt.data?.error);
                }
            }
            else if (isAborted && (lifecyclePhase === "end" || lifecyclePhase === "error")) {
                chatRunState.abortedRuns.delete(clientRunId);
                chatRunState.abortedRuns.delete(evt.runId);
                chatRunState.buffers.delete(clientRunId);
                chatRunState.deltaSentAt.delete(clientRunId);
                if (chatLink) {
                    chatRunState.registry.remove(evt.runId, clientRunId, sessionKey);
                }
            }
        }
        if (lifecyclePhase === "end" || lifecyclePhase === "error") {
            clearAgentRunContext(evt.runId);
        }
    };
}
