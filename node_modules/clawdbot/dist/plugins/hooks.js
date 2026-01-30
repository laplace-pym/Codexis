/**
 * Plugin Hook Runner
 *
 * Provides utilities for executing plugin lifecycle hooks with proper
 * error handling, priority ordering, and async support.
 */
/**
 * Get hooks for a specific hook name, sorted by priority (higher first).
 */
function getHooksForName(registry, hookName) {
    return registry.typedHooks
        .filter((h) => h.hookName === hookName)
        .sort((a, b) => (b.priority ?? 0) - (a.priority ?? 0));
}
/**
 * Create a hook runner for a specific registry.
 */
export function createHookRunner(registry, options = {}) {
    const logger = options.logger;
    const catchErrors = options.catchErrors ?? true;
    /**
     * Run a hook that doesn't return a value (fire-and-forget style).
     * All handlers are executed in parallel for performance.
     */
    async function runVoidHook(hookName, event, ctx) {
        const hooks = getHooksForName(registry, hookName);
        if (hooks.length === 0)
            return;
        logger?.debug?.(`[hooks] running ${hookName} (${hooks.length} handlers)`);
        const promises = hooks.map(async (hook) => {
            try {
                await hook.handler(event, ctx);
            }
            catch (err) {
                const msg = `[hooks] ${hookName} handler from ${hook.pluginId} failed: ${String(err)}`;
                if (catchErrors) {
                    logger?.error(msg);
                }
                else {
                    throw new Error(msg);
                }
            }
        });
        await Promise.all(promises);
    }
    /**
     * Run a hook that can return a modifying result.
     * Handlers are executed sequentially in priority order, and results are merged.
     */
    async function runModifyingHook(hookName, event, ctx, mergeResults) {
        const hooks = getHooksForName(registry, hookName);
        if (hooks.length === 0)
            return undefined;
        logger?.debug?.(`[hooks] running ${hookName} (${hooks.length} handlers, sequential)`);
        let result;
        for (const hook of hooks) {
            try {
                const handlerResult = await hook.handler(event, ctx);
                if (handlerResult !== undefined && handlerResult !== null) {
                    if (mergeResults && result !== undefined) {
                        result = mergeResults(result, handlerResult);
                    }
                    else {
                        result = handlerResult;
                    }
                }
            }
            catch (err) {
                const msg = `[hooks] ${hookName} handler from ${hook.pluginId} failed: ${String(err)}`;
                if (catchErrors) {
                    logger?.error(msg);
                }
                else {
                    throw new Error(msg);
                }
            }
        }
        return result;
    }
    // =========================================================================
    // Agent Hooks
    // =========================================================================
    /**
     * Run before_agent_start hook.
     * Allows plugins to inject context into the system prompt.
     * Runs sequentially, merging systemPrompt and prependContext from all handlers.
     */
    async function runBeforeAgentStart(event, ctx) {
        return runModifyingHook("before_agent_start", event, ctx, (acc, next) => ({
            systemPrompt: next.systemPrompt ?? acc?.systemPrompt,
            prependContext: acc?.prependContext && next.prependContext
                ? `${acc.prependContext}\n\n${next.prependContext}`
                : (next.prependContext ?? acc?.prependContext),
        }));
    }
    /**
     * Run agent_end hook.
     * Allows plugins to analyze completed conversations.
     * Runs in parallel (fire-and-forget).
     */
    async function runAgentEnd(event, ctx) {
        return runVoidHook("agent_end", event, ctx);
    }
    /**
     * Run before_compaction hook.
     */
    async function runBeforeCompaction(event, ctx) {
        return runVoidHook("before_compaction", event, ctx);
    }
    /**
     * Run after_compaction hook.
     */
    async function runAfterCompaction(event, ctx) {
        return runVoidHook("after_compaction", event, ctx);
    }
    // =========================================================================
    // Message Hooks
    // =========================================================================
    /**
     * Run message_received hook.
     * Runs in parallel (fire-and-forget).
     */
    async function runMessageReceived(event, ctx) {
        return runVoidHook("message_received", event, ctx);
    }
    /**
     * Run message_sending hook.
     * Allows plugins to modify or cancel outgoing messages.
     * Runs sequentially.
     */
    async function runMessageSending(event, ctx) {
        return runModifyingHook("message_sending", event, ctx, (acc, next) => ({
            content: next.content ?? acc?.content,
            cancel: next.cancel ?? acc?.cancel,
        }));
    }
    /**
     * Run message_sent hook.
     * Runs in parallel (fire-and-forget).
     */
    async function runMessageSent(event, ctx) {
        return runVoidHook("message_sent", event, ctx);
    }
    // =========================================================================
    // Tool Hooks
    // =========================================================================
    /**
     * Run before_tool_call hook.
     * Allows plugins to modify or block tool calls.
     * Runs sequentially.
     */
    async function runBeforeToolCall(event, ctx) {
        return runModifyingHook("before_tool_call", event, ctx, (acc, next) => ({
            params: next.params ?? acc?.params,
            block: next.block ?? acc?.block,
            blockReason: next.blockReason ?? acc?.blockReason,
        }));
    }
    /**
     * Run after_tool_call hook.
     * Runs in parallel (fire-and-forget).
     */
    async function runAfterToolCall(event, ctx) {
        return runVoidHook("after_tool_call", event, ctx);
    }
    /**
     * Run tool_result_persist hook.
     *
     * This hook is intentionally synchronous: it runs in hot paths where session
     * transcripts are appended synchronously.
     *
     * Handlers are executed sequentially in priority order (higher first). Each
     * handler may return `{ message }` to replace the message passed to the next
     * handler.
     */
    function runToolResultPersist(event, ctx) {
        const hooks = getHooksForName(registry, "tool_result_persist");
        if (hooks.length === 0)
            return undefined;
        let current = event.message;
        for (const hook of hooks) {
            try {
                const out = hook.handler({ ...event, message: current }, ctx);
                // Guard against accidental async handlers (this hook is sync-only).
                if (out && typeof out.then === "function") {
                    const msg = `[hooks] tool_result_persist handler from ${hook.pluginId} returned a Promise; ` +
                        `this hook is synchronous and the result was ignored.`;
                    if (catchErrors) {
                        logger?.warn?.(msg);
                        continue;
                    }
                    throw new Error(msg);
                }
                const next = out?.message;
                if (next)
                    current = next;
            }
            catch (err) {
                const msg = `[hooks] tool_result_persist handler from ${hook.pluginId} failed: ${String(err)}`;
                if (catchErrors) {
                    logger?.error(msg);
                }
                else {
                    throw new Error(msg);
                }
            }
        }
        return { message: current };
    }
    // =========================================================================
    // Session Hooks
    // =========================================================================
    /**
     * Run session_start hook.
     * Runs in parallel (fire-and-forget).
     */
    async function runSessionStart(event, ctx) {
        return runVoidHook("session_start", event, ctx);
    }
    /**
     * Run session_end hook.
     * Runs in parallel (fire-and-forget).
     */
    async function runSessionEnd(event, ctx) {
        return runVoidHook("session_end", event, ctx);
    }
    // =========================================================================
    // Gateway Hooks
    // =========================================================================
    /**
     * Run gateway_start hook.
     * Runs in parallel (fire-and-forget).
     */
    async function runGatewayStart(event, ctx) {
        return runVoidHook("gateway_start", event, ctx);
    }
    /**
     * Run gateway_stop hook.
     * Runs in parallel (fire-and-forget).
     */
    async function runGatewayStop(event, ctx) {
        return runVoidHook("gateway_stop", event, ctx);
    }
    // =========================================================================
    // Utility
    // =========================================================================
    /**
     * Check if any hooks are registered for a given hook name.
     */
    function hasHooks(hookName) {
        return registry.typedHooks.some((h) => h.hookName === hookName);
    }
    /**
     * Get count of registered hooks for a given hook name.
     */
    function getHookCount(hookName) {
        return registry.typedHooks.filter((h) => h.hookName === hookName).length;
    }
    return {
        // Agent hooks
        runBeforeAgentStart,
        runAgentEnd,
        runBeforeCompaction,
        runAfterCompaction,
        // Message hooks
        runMessageReceived,
        runMessageSending,
        runMessageSent,
        // Tool hooks
        runBeforeToolCall,
        runAfterToolCall,
        runToolResultPersist,
        // Session hooks
        runSessionStart,
        runSessionEnd,
        // Gateway hooks
        runGatewayStart,
        runGatewayStop,
        // Utility
        hasHooks,
        getHookCount,
    };
}
