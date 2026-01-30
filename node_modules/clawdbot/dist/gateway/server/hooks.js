import { randomUUID } from "node:crypto";
import { loadConfig } from "../../config/config.js";
import { resolveMainSessionKeyFromConfig } from "../../config/sessions.js";
import { runCronIsolatedAgentTurn } from "../../cron/isolated-agent.js";
import { requestHeartbeatNow } from "../../infra/heartbeat-wake.js";
import { enqueueSystemEvent } from "../../infra/system-events.js";
import { createHooksRequestHandler } from "../server-http.js";
export function createGatewayHooksRequestHandler(params) {
    const { deps, getHooksConfig, bindHost, port, logHooks } = params;
    const dispatchWakeHook = (value) => {
        const sessionKey = resolveMainSessionKeyFromConfig();
        enqueueSystemEvent(value.text, { sessionKey });
        if (value.mode === "now") {
            requestHeartbeatNow({ reason: "hook:wake" });
        }
    };
    const dispatchAgentHook = (value) => {
        const sessionKey = value.sessionKey.trim() ? value.sessionKey.trim() : `hook:${randomUUID()}`;
        const mainSessionKey = resolveMainSessionKeyFromConfig();
        const jobId = randomUUID();
        const now = Date.now();
        const job = {
            id: jobId,
            name: value.name,
            enabled: true,
            createdAtMs: now,
            updatedAtMs: now,
            schedule: { kind: "at", atMs: now },
            sessionTarget: "isolated",
            wakeMode: value.wakeMode,
            payload: {
                kind: "agentTurn",
                message: value.message,
                model: value.model,
                thinking: value.thinking,
                timeoutSeconds: value.timeoutSeconds,
                deliver: value.deliver,
                channel: value.channel,
                to: value.to,
            },
            state: { nextRunAtMs: now },
        };
        const runId = randomUUID();
        void (async () => {
            try {
                const cfg = loadConfig();
                const result = await runCronIsolatedAgentTurn({
                    cfg,
                    deps,
                    job,
                    message: value.message,
                    sessionKey,
                    lane: "cron",
                });
                const summary = result.summary?.trim() || result.error?.trim() || result.status;
                const prefix = result.status === "ok" ? `Hook ${value.name}` : `Hook ${value.name} (${result.status})`;
                enqueueSystemEvent(`${prefix}: ${summary}`.trim(), {
                    sessionKey: mainSessionKey,
                });
                if (value.wakeMode === "now") {
                    requestHeartbeatNow({ reason: `hook:${jobId}` });
                }
            }
            catch (err) {
                logHooks.warn(`hook agent failed: ${String(err)}`);
                enqueueSystemEvent(`Hook ${value.name} (error): ${String(err)}`, {
                    sessionKey: mainSessionKey,
                });
                if (value.wakeMode === "now") {
                    requestHeartbeatNow({ reason: `hook:${jobId}:error` });
                }
            }
        })();
        return runId;
    };
    return createHooksRequestHandler({
        getHooksConfig,
        bindHost,
        port,
        logHooks,
        dispatchAgentHook,
        dispatchWakeHook,
    });
}
