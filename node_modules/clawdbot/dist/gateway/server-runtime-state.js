import { WebSocketServer } from "ws";
import { CANVAS_HOST_PATH } from "../canvas-host/a2ui.js";
import { createCanvasHostHandler } from "../canvas-host/server.js";
import { createGatewayHooksRequestHandler } from "./server/hooks.js";
import { listenGatewayHttpServer } from "./server/http-listen.js";
import { resolveGatewayListenHosts } from "./net.js";
import { createGatewayPluginRequestHandler } from "./server/plugins-http.js";
import { createGatewayBroadcaster } from "./server-broadcast.js";
import { createChatRunState } from "./server-chat.js";
import { MAX_PAYLOAD_BYTES } from "./server-constants.js";
import { attachGatewayUpgradeHandler, createGatewayHttpServer } from "./server-http.js";
export async function createGatewayRuntimeState(params) {
    let canvasHost = null;
    if (params.canvasHostEnabled) {
        try {
            const handler = await createCanvasHostHandler({
                runtime: params.canvasRuntime,
                rootDir: params.cfg.canvasHost?.root,
                basePath: CANVAS_HOST_PATH,
                allowInTests: params.allowCanvasHostInTests,
                liveReload: params.cfg.canvasHost?.liveReload,
            });
            if (handler.rootDir) {
                canvasHost = handler;
                params.logCanvas.info(`canvas host mounted at http://${params.bindHost}:${params.port}${CANVAS_HOST_PATH}/ (root ${handler.rootDir})`);
            }
        }
        catch (err) {
            params.logCanvas.warn(`canvas host failed to start: ${String(err)}`);
        }
    }
    const handleHooksRequest = createGatewayHooksRequestHandler({
        deps: params.deps,
        getHooksConfig: params.hooksConfig,
        bindHost: params.bindHost,
        port: params.port,
        logHooks: params.logHooks,
    });
    const handlePluginRequest = createGatewayPluginRequestHandler({
        registry: params.pluginRegistry,
        log: params.logPlugins,
    });
    const bindHosts = await resolveGatewayListenHosts(params.bindHost);
    const httpServers = [];
    const httpBindHosts = [];
    for (const host of bindHosts) {
        const httpServer = createGatewayHttpServer({
            canvasHost,
            controlUiEnabled: params.controlUiEnabled,
            controlUiBasePath: params.controlUiBasePath,
            openAiChatCompletionsEnabled: params.openAiChatCompletionsEnabled,
            openResponsesEnabled: params.openResponsesEnabled,
            openResponsesConfig: params.openResponsesConfig,
            handleHooksRequest,
            handlePluginRequest,
            resolvedAuth: params.resolvedAuth,
            tlsOptions: params.gatewayTls?.enabled ? params.gatewayTls.tlsOptions : undefined,
        });
        try {
            await listenGatewayHttpServer({
                httpServer,
                bindHost: host,
                port: params.port,
            });
            httpServers.push(httpServer);
            httpBindHosts.push(host);
        }
        catch (err) {
            if (host === bindHosts[0])
                throw err;
            params.log.warn(`gateway: failed to bind loopback alias ${host}:${params.port} (${String(err)})`);
        }
    }
    const httpServer = httpServers[0];
    if (!httpServer) {
        throw new Error("Gateway HTTP server failed to start");
    }
    const wss = new WebSocketServer({
        noServer: true,
        maxPayload: MAX_PAYLOAD_BYTES,
    });
    for (const server of httpServers) {
        attachGatewayUpgradeHandler({ httpServer: server, wss, canvasHost });
    }
    const clients = new Set();
    const { broadcast } = createGatewayBroadcaster({ clients });
    const agentRunSeq = new Map();
    const dedupe = new Map();
    const chatRunState = createChatRunState();
    const chatRunRegistry = chatRunState.registry;
    const chatRunBuffers = chatRunState.buffers;
    const chatDeltaSentAt = chatRunState.deltaSentAt;
    const addChatRun = chatRunRegistry.add;
    const removeChatRun = chatRunRegistry.remove;
    const chatAbortControllers = new Map();
    return {
        canvasHost,
        httpServer,
        httpServers,
        httpBindHosts,
        wss,
        clients,
        broadcast,
        agentRunSeq,
        dedupe,
        chatRunState,
        chatRunBuffers,
        chatDeltaSentAt,
        addChatRun,
        removeChatRun,
        chatAbortControllers,
    };
}
