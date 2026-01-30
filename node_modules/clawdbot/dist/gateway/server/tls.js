import { loadGatewayTlsRuntime as loadGatewayTlsRuntimeConfig, } from "../../infra/tls/gateway.js";
export async function loadGatewayTlsRuntime(cfg, log) {
    return await loadGatewayTlsRuntimeConfig(cfg, log);
}
