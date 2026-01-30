import { createConfigIO, resolveConfigPath, resolveGatewayPort, resolveStateDir, } from "../../config/config.js";
import { readLastGatewayErrorLine } from "../../daemon/diagnostics.js";
import { findExtraGatewayServices } from "../../daemon/inspect.js";
import { findLegacyGatewayServices } from "../../daemon/legacy.js";
import { resolveGatewayService } from "../../daemon/service.js";
import { auditGatewayServiceConfig } from "../../daemon/service-audit.js";
import { resolveGatewayBindHost } from "../../gateway/net.js";
import { formatPortDiagnostics, inspectPortUsage, } from "../../infra/ports.js";
import { pickPrimaryTailnetIPv4 } from "../../infra/tailnet.js";
import { probeGatewayStatus } from "./probe.js";
import { normalizeListenerAddress, parsePortFromArgs, pickProbeHostForBind } from "./shared.js";
function shouldReportPortUsage(status, rpcOk) {
    if (status !== "busy")
        return false;
    if (rpcOk === true)
        return false;
    return true;
}
export async function gatherDaemonStatus(opts) {
    const service = resolveGatewayService();
    const [loaded, command, runtime] = await Promise.all([
        service.isLoaded({ env: process.env }).catch(() => false),
        service.readCommand(process.env).catch(() => null),
        service.readRuntime(process.env).catch((err) => ({ status: "unknown", detail: String(err) })),
    ]);
    const configAudit = await auditGatewayServiceConfig({
        env: process.env,
        command,
    });
    const serviceEnv = command?.environment ?? undefined;
    const mergedDaemonEnv = {
        ...process.env,
        ...(serviceEnv ?? undefined),
    };
    const cliConfigPath = resolveConfigPath(process.env, resolveStateDir(process.env));
    const daemonConfigPath = resolveConfigPath(mergedDaemonEnv, resolveStateDir(mergedDaemonEnv));
    const cliIO = createConfigIO({ env: process.env, configPath: cliConfigPath });
    const daemonIO = createConfigIO({
        env: mergedDaemonEnv,
        configPath: daemonConfigPath,
    });
    const [cliSnapshot, daemonSnapshot] = await Promise.all([
        cliIO.readConfigFileSnapshot().catch(() => null),
        daemonIO.readConfigFileSnapshot().catch(() => null),
    ]);
    const cliCfg = cliIO.loadConfig();
    const daemonCfg = daemonIO.loadConfig();
    const cliConfigSummary = {
        path: cliSnapshot?.path ?? cliConfigPath,
        exists: cliSnapshot?.exists ?? false,
        valid: cliSnapshot?.valid ?? true,
        ...(cliSnapshot?.issues?.length ? { issues: cliSnapshot.issues } : {}),
        controlUi: cliCfg.gateway?.controlUi,
    };
    const daemonConfigSummary = {
        path: daemonSnapshot?.path ?? daemonConfigPath,
        exists: daemonSnapshot?.exists ?? false,
        valid: daemonSnapshot?.valid ?? true,
        ...(daemonSnapshot?.issues?.length ? { issues: daemonSnapshot.issues } : {}),
        controlUi: daemonCfg.gateway?.controlUi,
    };
    const configMismatch = cliConfigSummary.path !== daemonConfigSummary.path;
    const portFromArgs = parsePortFromArgs(command?.programArguments);
    const daemonPort = portFromArgs ?? resolveGatewayPort(daemonCfg, mergedDaemonEnv);
    const portSource = portFromArgs
        ? "service args"
        : "env/config";
    const bindMode = (daemonCfg.gateway?.bind ?? "loopback");
    const customBindHost = daemonCfg.gateway?.customBindHost;
    const bindHost = await resolveGatewayBindHost(bindMode, customBindHost);
    const tailnetIPv4 = pickPrimaryTailnetIPv4();
    const probeHost = pickProbeHostForBind(bindMode, tailnetIPv4, customBindHost);
    const probeUrlOverride = typeof opts.rpc.url === "string" && opts.rpc.url.trim().length > 0 ? opts.rpc.url.trim() : null;
    const probeUrl = probeUrlOverride ?? `ws://${probeHost}:${daemonPort}`;
    const probeNote = !probeUrlOverride && bindMode === "lan"
        ? "Local probe uses loopback (127.0.0.1). bind=lan listens on 0.0.0.0 (all interfaces); use a LAN IP for remote clients."
        : !probeUrlOverride && bindMode === "loopback"
            ? "Loopback-only gateway; only local clients can connect."
            : undefined;
    const cliPort = resolveGatewayPort(cliCfg, process.env);
    const [portDiagnostics, portCliDiagnostics] = await Promise.all([
        inspectPortUsage(daemonPort).catch(() => null),
        cliPort !== daemonPort ? inspectPortUsage(cliPort).catch(() => null) : null,
    ]);
    const portStatus = portDiagnostics
        ? {
            port: portDiagnostics.port,
            status: portDiagnostics.status,
            listeners: portDiagnostics.listeners,
            hints: portDiagnostics.hints,
        }
        : undefined;
    const portCliStatus = portCliDiagnostics
        ? {
            port: portCliDiagnostics.port,
            status: portCliDiagnostics.status,
            listeners: portCliDiagnostics.listeners,
            hints: portCliDiagnostics.hints,
        }
        : undefined;
    const legacyServices = await findLegacyGatewayServices(process.env).catch(() => []);
    const extraServices = await findExtraGatewayServices(process.env, { deep: Boolean(opts.deep) }).catch(() => []);
    const timeoutMsRaw = Number.parseInt(String(opts.rpc.timeout ?? "10000"), 10);
    const timeoutMs = Number.isFinite(timeoutMsRaw) && timeoutMsRaw > 0 ? timeoutMsRaw : 10_000;
    const rpc = opts.probe
        ? await probeGatewayStatus({
            url: probeUrl,
            token: opts.rpc.token ||
                mergedDaemonEnv.CLAWDBOT_GATEWAY_TOKEN ||
                daemonCfg.gateway?.auth?.token,
            password: opts.rpc.password ||
                mergedDaemonEnv.CLAWDBOT_GATEWAY_PASSWORD ||
                daemonCfg.gateway?.auth?.password,
            timeoutMs,
            json: opts.rpc.json,
            configPath: daemonConfigSummary.path,
        })
        : undefined;
    let lastError;
    if (loaded && runtime?.status === "running" && portStatus && portStatus.status !== "busy") {
        lastError = (await readLastGatewayErrorLine(mergedDaemonEnv)) ?? undefined;
    }
    return {
        service: {
            label: service.label,
            loaded,
            loadedText: service.loadedText,
            notLoadedText: service.notLoadedText,
            command,
            runtime,
            configAudit,
        },
        config: {
            cli: cliConfigSummary,
            daemon: daemonConfigSummary,
            ...(configMismatch ? { mismatch: true } : {}),
        },
        gateway: {
            bindMode,
            bindHost,
            customBindHost,
            port: daemonPort,
            portSource,
            probeUrl,
            ...(probeNote ? { probeNote } : {}),
        },
        port: portStatus,
        ...(portCliStatus ? { portCli: portCliStatus } : {}),
        lastError,
        ...(rpc ? { rpc: { ...rpc, url: probeUrl } } : {}),
        legacyServices,
        extraServices,
    };
}
export function renderPortDiagnosticsForCli(status, rpcOk) {
    if (!status.port || !shouldReportPortUsage(status.port.status, rpcOk))
        return [];
    return formatPortDiagnostics({
        port: status.port.port,
        status: status.port.status,
        listeners: status.port.listeners,
        hints: status.port.hints,
    });
}
export function resolvePortListeningAddresses(status) {
    const addrs = Array.from(new Set(status.port?.listeners
        ?.map((l) => (l.address ? normalizeListenerAddress(l.address) : ""))
        .filter((v) => Boolean(v)) ?? []));
    return addrs;
}
