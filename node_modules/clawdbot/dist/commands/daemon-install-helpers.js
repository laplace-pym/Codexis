import { resolveGatewayLaunchAgentLabel } from "../daemon/constants.js";
import { resolveGatewayProgramArguments } from "../daemon/program-args.js";
import { renderSystemNodeWarning, resolvePreferredNodePath, resolveSystemNodeInfo, } from "../daemon/runtime-paths.js";
import { buildServiceEnvironment } from "../daemon/service-env.js";
import { formatCliCommand } from "../cli/command-format.js";
import { collectConfigEnvVars } from "../config/env-vars.js";
export function resolveGatewayDevMode(argv = process.argv) {
    const entry = argv[1];
    const normalizedEntry = entry?.replaceAll("\\", "/");
    return Boolean(normalizedEntry?.includes("/src/") && normalizedEntry.endsWith(".ts"));
}
export async function buildGatewayInstallPlan(params) {
    const devMode = params.devMode ?? resolveGatewayDevMode();
    const nodePath = params.nodePath ??
        (await resolvePreferredNodePath({
            env: params.env,
            runtime: params.runtime,
        }));
    const { programArguments, workingDirectory } = await resolveGatewayProgramArguments({
        port: params.port,
        dev: devMode,
        runtime: params.runtime,
        nodePath,
    });
    if (params.runtime === "node") {
        const systemNode = await resolveSystemNodeInfo({ env: params.env });
        const warning = renderSystemNodeWarning(systemNode, programArguments[0]);
        if (warning)
            params.warn?.(warning, "Gateway runtime");
    }
    const serviceEnvironment = buildServiceEnvironment({
        env: params.env,
        port: params.port,
        token: params.token,
        launchdLabel: process.platform === "darwin"
            ? resolveGatewayLaunchAgentLabel(params.env.CLAWDBOT_PROFILE)
            : undefined,
    });
    // Merge config env vars into the service environment (vars + inline env keys).
    // Config env vars are added first so service-specific vars take precedence.
    const environment = {
        ...collectConfigEnvVars(params.config),
    };
    Object.assign(environment, serviceEnvironment);
    return { programArguments, workingDirectory, environment };
}
export function gatewayInstallErrorHint(platform = process.platform) {
    return platform === "win32"
        ? "Tip: rerun from an elevated PowerShell (Start → type PowerShell → right-click → Run as administrator) or skip service install."
        : `Tip: rerun \`${formatCliCommand("clawdbot gateway install")}\` after fixing the error.`;
}
