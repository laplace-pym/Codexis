import express from "express";
import { registerBrowserRoutes } from "./routes/index.js";
import { createBrowserRouteContext, } from "./server-context.js";
export async function startBrowserBridgeServer(params) {
    const host = params.host ?? "127.0.0.1";
    const port = params.port ?? 0;
    const app = express();
    app.use(express.json({ limit: "1mb" }));
    const authToken = params.authToken?.trim();
    if (authToken) {
        app.use((req, res, next) => {
            const auth = String(req.headers.authorization ?? "").trim();
            if (auth === `Bearer ${authToken}`)
                return next();
            res.status(401).send("Unauthorized");
        });
    }
    const state = {
        server: null,
        port,
        resolved: params.resolved,
        profiles: new Map(),
    };
    const ctx = createBrowserRouteContext({
        getState: () => state,
        onEnsureAttachTarget: params.onEnsureAttachTarget,
    });
    registerBrowserRoutes(app, ctx);
    const server = await new Promise((resolve, reject) => {
        const s = app.listen(port, host, () => resolve(s));
        s.once("error", reject);
    });
    const address = server.address();
    const resolvedPort = address?.port ?? port;
    state.server = server;
    state.port = resolvedPort;
    state.resolved.controlHost = host;
    state.resolved.controlPort = resolvedPort;
    state.resolved.controlUrl = `http://${host}:${resolvedPort}`;
    const baseUrl = state.resolved.controlUrl;
    return { server, port: resolvedPort, baseUrl, state };
}
export async function stopBrowserBridgeServer(server) {
    await new Promise((resolve) => {
        server.close(() => resolve());
    });
}
