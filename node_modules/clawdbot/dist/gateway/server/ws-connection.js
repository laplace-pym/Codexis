import { randomUUID } from "node:crypto";
import { resolveCanvasHostUrl } from "../../infra/canvas-host-url.js";
import { listSystemPresence, upsertPresence } from "../../infra/system-presence.js";
import { isWebchatClient } from "../../utils/message-channel.js";
import { isLoopbackAddress } from "../net.js";
import { getHandshakeTimeoutMs } from "../server-constants.js";
import { formatError } from "../server-utils.js";
import { logWs } from "../ws-log.js";
import { getHealthVersion, getPresenceVersion, incrementPresenceVersion } from "./health-state.js";
import { attachGatewayWsMessageHandler } from "./ws-connection/message-handler.js";
export function attachGatewayWsConnectionHandler(params) {
    const { wss, clients, port, gatewayHost, canvasHostEnabled, canvasHostServerPort, resolvedAuth, gatewayMethods, events, logGateway, logHealth, logWsControl, extraHandlers, broadcast, buildRequestContext, } = params;
    wss.on("connection", (socket, upgradeReq) => {
        let client = null;
        let closed = false;
        const openedAt = Date.now();
        const connId = randomUUID();
        const remoteAddr = socket._socket
            ?.remoteAddress;
        const headerValue = (value) => Array.isArray(value) ? value[0] : value;
        const requestHost = headerValue(upgradeReq.headers.host);
        const requestOrigin = headerValue(upgradeReq.headers.origin);
        const requestUserAgent = headerValue(upgradeReq.headers["user-agent"]);
        const forwardedFor = headerValue(upgradeReq.headers["x-forwarded-for"]);
        const realIp = headerValue(upgradeReq.headers["x-real-ip"]);
        const canvasHostPortForWs = canvasHostServerPort ?? (canvasHostEnabled ? port : undefined);
        const canvasHostOverride = gatewayHost && gatewayHost !== "0.0.0.0" && gatewayHost !== "::" ? gatewayHost : undefined;
        const canvasHostUrl = resolveCanvasHostUrl({
            canvasPort: canvasHostPortForWs,
            hostOverride: canvasHostServerPort ? canvasHostOverride : undefined,
            requestHost: upgradeReq.headers.host,
            forwardedProto: upgradeReq.headers["x-forwarded-proto"],
            localAddress: upgradeReq.socket?.localAddress,
        });
        logWs("in", "open", { connId, remoteAddr });
        let handshakeState = "pending";
        let closeCause;
        let closeMeta = {};
        let lastFrameType;
        let lastFrameMethod;
        let lastFrameId;
        const setCloseCause = (cause, meta) => {
            if (!closeCause)
                closeCause = cause;
            if (meta && Object.keys(meta).length > 0) {
                closeMeta = { ...closeMeta, ...meta };
            }
        };
        const setLastFrameMeta = (meta) => {
            if (meta.type || meta.method || meta.id) {
                lastFrameType = meta.type ?? lastFrameType;
                lastFrameMethod = meta.method ?? lastFrameMethod;
                lastFrameId = meta.id ?? lastFrameId;
            }
        };
        const send = (obj) => {
            try {
                socket.send(JSON.stringify(obj));
            }
            catch {
                /* ignore */
            }
        };
        const connectNonce = randomUUID();
        send({
            type: "event",
            event: "connect.challenge",
            payload: { nonce: connectNonce, ts: Date.now() },
        });
        const close = (code = 1000, reason) => {
            if (closed)
                return;
            closed = true;
            clearTimeout(handshakeTimer);
            if (client)
                clients.delete(client);
            try {
                socket.close(code, reason);
            }
            catch {
                /* ignore */
            }
        };
        socket.once("error", (err) => {
            logWsControl.warn(`error conn=${connId} remote=${remoteAddr ?? "?"}: ${formatError(err)}`);
            close();
        });
        const isNoisySwiftPmHelperClose = (userAgent, remote) => Boolean(userAgent?.toLowerCase().includes("swiftpm-testing-helper") && isLoopbackAddress(remote));
        socket.once("close", (code, reason) => {
            const durationMs = Date.now() - openedAt;
            const closeContext = {
                cause: closeCause,
                handshake: handshakeState,
                durationMs,
                lastFrameType,
                lastFrameMethod,
                lastFrameId,
                host: requestHost,
                origin: requestOrigin,
                userAgent: requestUserAgent,
                forwardedFor,
                ...closeMeta,
            };
            if (!client) {
                const logFn = isNoisySwiftPmHelperClose(requestUserAgent, remoteAddr)
                    ? logWsControl.debug
                    : logWsControl.warn;
                logFn(`closed before connect conn=${connId} remote=${remoteAddr ?? "?"} fwd=${forwardedFor ?? "n/a"} origin=${requestOrigin ?? "n/a"} host=${requestHost ?? "n/a"} ua=${requestUserAgent ?? "n/a"} code=${code ?? "n/a"} reason=${reason?.toString() || "n/a"}`, closeContext);
            }
            if (client && isWebchatClient(client.connect.client)) {
                logWsControl.info(`webchat disconnected code=${code} reason=${reason?.toString() || "n/a"} conn=${connId}`);
            }
            if (client?.presenceKey) {
                upsertPresence(client.presenceKey, { reason: "disconnect" });
                incrementPresenceVersion();
                broadcast("presence", { presence: listSystemPresence() }, {
                    dropIfSlow: true,
                    stateVersion: {
                        presence: getPresenceVersion(),
                        health: getHealthVersion(),
                    },
                });
            }
            if (client?.connect?.role === "node") {
                const context = buildRequestContext();
                const nodeId = context.nodeRegistry.unregister(connId);
                if (nodeId) {
                    context.nodeUnsubscribeAll(nodeId);
                }
            }
            logWs("out", "close", {
                connId,
                code,
                reason: reason?.toString(),
                durationMs,
                cause: closeCause,
                handshake: handshakeState,
                lastFrameType,
                lastFrameMethod,
                lastFrameId,
            });
            close();
        });
        const handshakeTimeoutMs = getHandshakeTimeoutMs();
        const handshakeTimer = setTimeout(() => {
            if (!client) {
                handshakeState = "failed";
                setCloseCause("handshake-timeout", {
                    handshakeMs: Date.now() - openedAt,
                });
                logWsControl.warn(`handshake timeout conn=${connId} remote=${remoteAddr ?? "?"}`);
                close();
            }
        }, handshakeTimeoutMs);
        attachGatewayWsMessageHandler({
            socket,
            upgradeReq,
            connId,
            remoteAddr,
            forwardedFor,
            realIp,
            requestHost,
            requestOrigin,
            requestUserAgent,
            canvasHostUrl,
            connectNonce,
            resolvedAuth,
            gatewayMethods,
            events,
            extraHandlers,
            buildRequestContext,
            send,
            close,
            isClosed: () => closed,
            clearHandshakeTimer: () => clearTimeout(handshakeTimer),
            getClient: () => client,
            setClient: (next) => {
                client = next;
                clients.add(next);
            },
            setHandshakeState: (next) => {
                handshakeState = next;
            },
            setCloseCause,
            setLastFrameMeta,
            logGateway,
            logHealth,
            logWsControl,
        });
    });
}
