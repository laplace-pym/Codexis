import { resolveGatewayService } from "../daemon/service.js";
import { resolveNodeService } from "../daemon/node-service.js";
import { formatDaemonRuntimeShort } from "./status.format.js";
async function buildDaemonStatusSummary(service, fallbackLabel) {
    try {
        const [loaded, runtime, command] = await Promise.all([
            service.isLoaded({ env: process.env }).catch(() => false),
            service.readRuntime(process.env).catch(() => undefined),
            service.readCommand(process.env).catch(() => null),
        ]);
        const installed = command != null;
        const loadedText = loaded ? service.loadedText : service.notLoadedText;
        const runtimeShort = formatDaemonRuntimeShort(runtime);
        return { label: service.label, installed, loadedText, runtimeShort };
    }
    catch {
        return {
            label: fallbackLabel,
            installed: null,
            loadedText: "unknown",
            runtimeShort: null,
        };
    }
}
export async function getDaemonStatusSummary() {
    return await buildDaemonStatusSummary(resolveGatewayService(), "Daemon");
}
export async function getNodeDaemonStatusSummary() {
    return await buildDaemonStatusSummary(resolveNodeService(), "Node");
}
