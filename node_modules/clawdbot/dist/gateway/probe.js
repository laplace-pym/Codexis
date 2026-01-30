import { randomUUID } from "node:crypto";
import { GATEWAY_CLIENT_MODES, GATEWAY_CLIENT_NAMES } from "../utils/message-channel.js";
import { GatewayClient } from "./client.js";
function formatError(err) {
    if (err instanceof Error)
        return err.message;
    return String(err);
}
export async function probeGateway(opts) {
    const startedAt = Date.now();
    const instanceId = randomUUID();
    let connectLatencyMs = null;
    let connectError = null;
    let close = null;
    return await new Promise((resolve) => {
        let settled = false;
        const settle = (result) => {
            if (settled)
                return;
            settled = true;
            clearTimeout(timer);
            client.stop();
            resolve({ url: opts.url, ...result });
        };
        const client = new GatewayClient({
            url: opts.url,
            token: opts.auth?.token,
            password: opts.auth?.password,
            clientName: GATEWAY_CLIENT_NAMES.CLI,
            clientVersion: "dev",
            mode: GATEWAY_CLIENT_MODES.PROBE,
            instanceId,
            onConnectError: (err) => {
                connectError = formatError(err);
            },
            onClose: (code, reason) => {
                close = { code, reason };
            },
            onHelloOk: async () => {
                connectLatencyMs = Date.now() - startedAt;
                try {
                    const [health, status, presence, configSnapshot] = await Promise.all([
                        client.request("health"),
                        client.request("status"),
                        client.request("system-presence"),
                        client.request("config.get", {}),
                    ]);
                    settle({
                        ok: true,
                        connectLatencyMs,
                        error: null,
                        close,
                        health,
                        status,
                        presence: Array.isArray(presence) ? presence : null,
                        configSnapshot,
                    });
                }
                catch (err) {
                    settle({
                        ok: false,
                        connectLatencyMs,
                        error: formatError(err),
                        close,
                        health: null,
                        status: null,
                        presence: null,
                        configSnapshot: null,
                    });
                }
            },
        });
        const timer = setTimeout(() => {
            settle({
                ok: false,
                connectLatencyMs,
                error: connectError ? `connect failed: ${connectError}` : "timeout",
                close,
                health: null,
                status: null,
                presence: null,
                configSnapshot: null,
            });
        }, Math.max(250, opts.timeoutMs));
        client.start();
    });
}
