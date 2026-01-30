import { makeMissingToolResult } from "./session-transcript-repair.js";
import { emitSessionTranscriptUpdate } from "../sessions/transcript-events.js";
function extractAssistantToolCalls(msg) {
    const content = msg.content;
    if (!Array.isArray(content))
        return [];
    const toolCalls = [];
    for (const block of content) {
        if (!block || typeof block !== "object")
            continue;
        const rec = block;
        if (typeof rec.id !== "string" || !rec.id)
            continue;
        if (rec.type === "toolCall" || rec.type === "toolUse" || rec.type === "functionCall") {
            toolCalls.push({
                id: rec.id,
                name: typeof rec.name === "string" ? rec.name : undefined,
            });
        }
    }
    return toolCalls;
}
function extractToolResultId(msg) {
    const toolCallId = msg.toolCallId;
    if (typeof toolCallId === "string" && toolCallId)
        return toolCallId;
    const toolUseId = msg.toolUseId;
    if (typeof toolUseId === "string" && toolUseId)
        return toolUseId;
    return null;
}
export function installSessionToolResultGuard(sessionManager, opts) {
    const originalAppend = sessionManager.appendMessage.bind(sessionManager);
    const pending = new Map();
    const persistToolResult = (message, meta) => {
        const transformer = opts?.transformToolResultForPersistence;
        return transformer ? transformer(message, meta) : message;
    };
    const allowSyntheticToolResults = opts?.allowSyntheticToolResults ?? true;
    const flushPendingToolResults = () => {
        if (pending.size === 0)
            return;
        if (allowSyntheticToolResults) {
            for (const [id, name] of pending.entries()) {
                const synthetic = makeMissingToolResult({ toolCallId: id, toolName: name });
                originalAppend(persistToolResult(synthetic, {
                    toolCallId: id,
                    toolName: name,
                    isSynthetic: true,
                }));
            }
        }
        pending.clear();
    };
    const guardedAppend = (message) => {
        const role = message.role;
        if (role === "toolResult") {
            const id = extractToolResultId(message);
            const toolName = id ? pending.get(id) : undefined;
            if (id)
                pending.delete(id);
            return originalAppend(persistToolResult(message, {
                toolCallId: id ?? undefined,
                toolName,
                isSynthetic: false,
            }));
        }
        const toolCalls = role === "assistant"
            ? extractAssistantToolCalls(message)
            : [];
        if (allowSyntheticToolResults) {
            // If previous tool calls are still pending, flush before non-tool results.
            if (pending.size > 0 && (toolCalls.length === 0 || role !== "assistant")) {
                flushPendingToolResults();
            }
            // If new tool calls arrive while older ones are pending, flush the old ones first.
            if (pending.size > 0 && toolCalls.length > 0) {
                flushPendingToolResults();
            }
        }
        const result = originalAppend(message);
        const sessionFile = sessionManager.getSessionFile?.();
        if (sessionFile) {
            emitSessionTranscriptUpdate(sessionFile);
        }
        if (toolCalls.length > 0) {
            for (const call of toolCalls) {
                pending.set(call.id, call.name);
            }
        }
        return result;
    };
    // Monkey-patch appendMessage with our guarded version.
    sessionManager.appendMessage = guardedAppend;
    return {
        flushPendingToolResults,
        getPendingIds: () => Array.from(pending.keys()),
    };
}
