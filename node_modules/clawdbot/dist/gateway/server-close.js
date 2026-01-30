import { listChannelPlugins } from "../channels/plugins/index.js";
import { stopGmailWatcher } from "../hooks/gmail-watcher.js";
export function createGatewayCloseHandler(params) {
    return async (opts) => {
        const reasonRaw = typeof opts?.reason === "string" ? opts.reason.trim() : "";
        const reason = reasonRaw || "gateway stopping";
        const restartExpectedMs = typeof opts?.restartExpectedMs === "number" && Number.isFinite(opts.restartExpectedMs)
            ? Math.max(0, Math.floor(opts.restartExpectedMs))
            : null;
        if (params.bonjourStop) {
            try {
                await params.bonjourStop();
            }
            catch {
                /* ignore */
            }
        }
        if (params.tailscaleCleanup) {
            await params.tailscaleCleanup();
        }
        if (params.canvasHost) {
            try {
                await params.canvasHost.close();
            }
            catch {
                /* ignore */
            }
        }
        if (params.canvasHostServer) {
            try {
                await params.canvasHostServer.close();
            }
            catch {
                /* ignore */
            }
        }
        for (const plugin of listChannelPlugins()) {
            await params.stopChannel(plugin.id);
        }
        if (params.pluginServices) {
            await params.pluginServices.stop().catch(() => { });
        }
        await stopGmailWatcher();
        params.cron.stop();
        params.heartbeatRunner.stop();
        for (const timer of params.nodePresenceTimers.values()) {
            clearInterval(timer);
        }
        params.nodePresenceTimers.clear();
        params.broadcast("shutdown", {
            reason,
            restartExpectedMs,
        });
        clearInterval(params.tickInterval);
        clearInterval(params.healthInterval);
        clearInterval(params.dedupeCleanup);
        if (params.agentUnsub) {
            try {
                params.agentUnsub();
            }
            catch {
                /* ignore */
            }
        }
        if (params.heartbeatUnsub) {
            try {
                params.heartbeatUnsub();
            }
            catch {
                /* ignore */
            }
        }
        params.chatRunState.clear();
        for (const c of params.clients) {
            try {
                c.socket.close(1012, "service restart");
            }
            catch {
                /* ignore */
            }
        }
        params.clients.clear();
        await params.configReloader.stop().catch(() => { });
        if (params.browserControl) {
            await params.browserControl.stop().catch(() => { });
        }
        await new Promise((resolve) => params.wss.close(() => resolve()));
        const servers = params.httpServers && params.httpServers.length > 0
            ? params.httpServers
            : [params.httpServer];
        for (const server of servers) {
            const httpServer = server;
            if (typeof httpServer.closeIdleConnections === "function") {
                httpServer.closeIdleConnections();
            }
            await new Promise((resolve, reject) => httpServer.close((err) => (err ? reject(err) : resolve())));
        }
    };
}
