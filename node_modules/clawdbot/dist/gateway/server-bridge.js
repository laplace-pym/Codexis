import { ErrorCodes } from "./protocol/index.js";
import { handleBridgeEvent as handleBridgeEventImpl } from "./server-bridge-events.js";
import { handleChatBridgeMethods } from "./server-bridge-methods-chat.js";
import { handleConfigBridgeMethods } from "./server-bridge-methods-config.js";
import { handleSessionsBridgeMethods } from "./server-bridge-methods-sessions.js";
import { handleSystemBridgeMethods } from "./server-bridge-methods-system.js";
export function createBridgeHandlers(ctx) {
    const handleBridgeRequest = async (nodeId, req) => {
        const method = req.method.trim();
        const parseParams = () => {
            const raw = typeof req.paramsJSON === "string" ? req.paramsJSON : "";
            const trimmed = raw.trim();
            if (!trimmed)
                return {};
            const parsed = JSON.parse(trimmed);
            return typeof parsed === "object" && parsed !== null
                ? parsed
                : {};
        };
        try {
            const params = parseParams();
            const response = (await handleSystemBridgeMethods(ctx, nodeId, method, params)) ??
                (await handleConfigBridgeMethods(ctx, nodeId, method, params)) ??
                (await handleSessionsBridgeMethods(ctx, nodeId, method, params)) ??
                (await handleChatBridgeMethods(ctx, nodeId, method, params));
            if (response)
                return response;
            return {
                ok: false,
                error: {
                    code: "FORBIDDEN",
                    message: "Method not allowed",
                    details: { method },
                },
            };
        }
        catch (err) {
            return {
                ok: false,
                error: { code: ErrorCodes.INVALID_REQUEST, message: String(err) },
            };
        }
    };
    const handleBridgeEvent = async (nodeId, evt) => {
        await handleBridgeEventImpl(ctx, nodeId, evt);
    };
    return { handleBridgeRequest, handleBridgeEvent };
}
