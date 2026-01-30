import { attachGatewayWsConnectionHandler } from "./server/ws-connection.js";
export function attachGatewayWsHandlers(params) {
    attachGatewayWsConnectionHandler({
        wss: params.wss,
        clients: params.clients,
        port: params.port,
        gatewayHost: params.gatewayHost,
        canvasHostEnabled: params.canvasHostEnabled,
        canvasHostServerPort: params.canvasHostServerPort,
        resolvedAuth: params.resolvedAuth,
        gatewayMethods: params.gatewayMethods,
        events: params.events,
        logGateway: params.logGateway,
        logHealth: params.logHealth,
        logWsControl: params.logWsControl,
        extraHandlers: params.extraHandlers,
        broadcast: params.broadcast,
        buildRequestContext: () => params.context,
    });
}
