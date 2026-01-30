import { isAbortTrigger } from "../auto-reply/reply/abort.js";
export function isChatStopCommandText(text) {
    const trimmed = text.trim();
    if (!trimmed)
        return false;
    return trimmed.toLowerCase() === "/stop" || isAbortTrigger(trimmed);
}
export function resolveChatRunExpiresAtMs(params) {
    const { now, timeoutMs, graceMs = 60_000, minMs = 2 * 60_000, maxMs = 24 * 60 * 60_000 } = params;
    const boundedTimeoutMs = Math.max(0, timeoutMs);
    const target = now + boundedTimeoutMs + graceMs;
    const min = now + minMs;
    const max = now + maxMs;
    return Math.min(max, Math.max(min, target));
}
function broadcastChatAborted(ops, params) {
    const { runId, sessionKey, stopReason } = params;
    const payload = {
        runId,
        sessionKey,
        seq: (ops.agentRunSeq.get(runId) ?? 0) + 1,
        state: "aborted",
        stopReason,
    };
    ops.broadcast("chat", payload);
    ops.nodeSendToSession(sessionKey, "chat", payload);
}
export function abortChatRunById(ops, params) {
    const { runId, sessionKey, stopReason } = params;
    const active = ops.chatAbortControllers.get(runId);
    if (!active)
        return { aborted: false };
    if (active.sessionKey !== sessionKey)
        return { aborted: false };
    ops.chatAbortedRuns.set(runId, Date.now());
    active.controller.abort();
    ops.chatAbortControllers.delete(runId);
    ops.chatRunBuffers.delete(runId);
    ops.chatDeltaSentAt.delete(runId);
    ops.removeChatRun(runId, runId, sessionKey);
    broadcastChatAborted(ops, { runId, sessionKey, stopReason });
    return { aborted: true };
}
export function abortChatRunsForSessionKey(ops, params) {
    const { sessionKey, stopReason } = params;
    const runIds = [];
    for (const [runId, active] of ops.chatAbortControllers) {
        if (active.sessionKey !== sessionKey)
            continue;
        const res = abortChatRunById(ops, { runId, sessionKey, stopReason });
        if (res.aborted)
            runIds.push(runId);
    }
    return { aborted: runIds.length > 0, runIds };
}
