import { getGlobalHookRunner } from "../plugins/hook-runner-global.js";
import { installSessionToolResultGuard } from "./session-tool-result-guard.js";
/**
 * Apply the tool-result guard to a SessionManager exactly once and expose
 * a flush method on the instance for easy teardown handling.
 */
export function guardSessionManager(sessionManager, opts) {
    if (typeof sessionManager.flushPendingToolResults === "function") {
        return sessionManager;
    }
    const hookRunner = getGlobalHookRunner();
    const transform = hookRunner?.hasHooks("tool_result_persist")
        ? (message, meta) => {
            const out = hookRunner.runToolResultPersist({
                toolName: meta.toolName,
                toolCallId: meta.toolCallId,
                message,
                isSynthetic: meta.isSynthetic,
            }, {
                agentId: opts?.agentId,
                sessionKey: opts?.sessionKey,
                toolName: meta.toolName,
                toolCallId: meta.toolCallId,
            });
            return out?.message ?? message;
        }
        : undefined;
    const guard = installSessionToolResultGuard(sessionManager, {
        transformToolResultForPersistence: transform,
        allowSyntheticToolResults: opts?.allowSyntheticToolResults,
    });
    sessionManager.flushPendingToolResults = guard.flushPendingToolResults;
    return sessionManager;
}
