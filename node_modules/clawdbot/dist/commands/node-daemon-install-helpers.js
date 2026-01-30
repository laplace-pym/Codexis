import { formatNodeServiceDescription } from "../daemon/constants.js";
import { resolveNodeProgramArguments } from "../daemon/program-args.js";
import { renderSystemNodeWarning, resolvePreferredNodePath, resolveSystemNodeInfo, } from "../daemon/runtime-paths.js";
import { buildNodeServiceEnvironment } from "../daemon/service-env.js";
import { resolveGatewayDevMode } from "./daemon-install-helpers.js";
export async function buildNodeInstallPlan(params) {
    const devMode = params.devMode ?? resolveGatewayDevMode();
    const nodePath = params.nodePath ??
        (await resolvePreferredNodePath({
            env: params.env,
            runtime: params.runtime,
        }));
    const { programArguments, workingDirectory } = await resolveNodeProgramArguments({
        host: params.host,
        port: params.port,
        tls: params.tls,
        tlsFingerprint: params.tlsFingerprint,
        nodeId: params.nodeId,
        displayName: params.displayName,
        dev: devMode,
        runtime: params.runtime,
        nodePath,
    });
    if (params.runtime === "node") {
        const systemNode = await resolveSystemNodeInfo({ env: params.env });
        const warning = renderSystemNodeWarning(systemNode, programArguments[0]);
        if (warning)
            params.warn?.(warning, "Node daemon runtime");
    }
    const environment = buildNodeServiceEnvironment({ env: params.env });
    const description = formatNodeServiceDescription({
        version: environment.CLAWDBOT_SERVICE_VERSION,
    });
    return { programArguments, workingDirectory, environment, description };
}
