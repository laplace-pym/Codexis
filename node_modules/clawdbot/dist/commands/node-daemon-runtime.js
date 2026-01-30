import { DEFAULT_GATEWAY_DAEMON_RUNTIME, GATEWAY_DAEMON_RUNTIME_OPTIONS, isGatewayDaemonRuntime, } from "./daemon-runtime.js";
export const DEFAULT_NODE_DAEMON_RUNTIME = DEFAULT_GATEWAY_DAEMON_RUNTIME;
export const NODE_DAEMON_RUNTIME_OPTIONS = GATEWAY_DAEMON_RUNTIME_OPTIONS;
export function isNodeDaemonRuntime(value) {
    return isGatewayDaemonRuntime(value);
}
