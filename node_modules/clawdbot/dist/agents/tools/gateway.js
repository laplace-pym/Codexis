import { callGateway } from "../../gateway/call.js";
import { GATEWAY_CLIENT_MODES, GATEWAY_CLIENT_NAMES } from "../../utils/message-channel.js";
export const DEFAULT_GATEWAY_URL = "ws://127.0.0.1:18789";
export function resolveGatewayOptions(opts) {
    // Prefer an explicit override; otherwise let callGateway choose based on config.
    const url = typeof opts?.gatewayUrl === "string" && opts.gatewayUrl.trim()
        ? opts.gatewayUrl.trim()
        : undefined;
    const token = typeof opts?.gatewayToken === "string" && opts.gatewayToken.trim()
        ? opts.gatewayToken.trim()
        : undefined;
    const timeoutMs = typeof opts?.timeoutMs === "number" && Number.isFinite(opts.timeoutMs)
        ? Math.max(1, Math.floor(opts.timeoutMs))
        : 10_000;
    return { url, token, timeoutMs };
}
export async function callGatewayTool(method, opts, params, extra) {
    const gateway = resolveGatewayOptions(opts);
    return await callGateway({
        url: gateway.url,
        token: gateway.token,
        method,
        params,
        timeoutMs: gateway.timeoutMs,
        expectFinal: extra?.expectFinal,
        clientName: GATEWAY_CLIENT_NAMES.GATEWAY_CLIENT,
        clientDisplayName: "agent",
        mode: GATEWAY_CLIENT_MODES.BACKEND,
    });
}
