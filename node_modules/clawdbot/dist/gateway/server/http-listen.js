import { GatewayLockError } from "../../infra/gateway-lock.js";
export async function listenGatewayHttpServer(params) {
    const { httpServer, bindHost, port } = params;
    try {
        await new Promise((resolve, reject) => {
            const onError = (err) => {
                httpServer.off("listening", onListening);
                reject(err);
            };
            const onListening = () => {
                httpServer.off("error", onError);
                resolve();
            };
            httpServer.once("error", onError);
            httpServer.once("listening", onListening);
            httpServer.listen(port, bindHost);
        });
    }
    catch (err) {
        const code = err.code;
        if (code === "EADDRINUSE") {
            throw new GatewayLockError(`another gateway instance is already listening on ws://${bindHost}:${port}`, err);
        }
        throw new GatewayLockError(`failed to bind gateway socket on ws://${bindHost}:${port}: ${String(err)}`, err);
    }
}
