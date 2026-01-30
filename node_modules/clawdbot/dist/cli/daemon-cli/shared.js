import { resolveGatewayLaunchAgentLabel, resolveGatewaySystemdServiceName, resolveGatewayWindowsTaskName, } from "../../daemon/constants.js";
import { resolveGatewayLogPaths } from "../../daemon/launchd.js";
import { getResolvedLoggerSettings } from "../../logging.js";
import { formatCliCommand } from "../command-format.js";
export function parsePort(raw) {
    if (raw === undefined || raw === null)
        return null;
    const value = typeof raw === "string"
        ? raw
        : typeof raw === "number" || typeof raw === "bigint"
            ? raw.toString()
            : null;
    if (value === null)
        return null;
    const parsed = Number.parseInt(value, 10);
    if (!Number.isFinite(parsed) || parsed <= 0)
        return null;
    return parsed;
}
export function parsePortFromArgs(programArguments) {
    if (!programArguments?.length)
        return null;
    for (let i = 0; i < programArguments.length; i += 1) {
        const arg = programArguments[i];
        if (arg === "--port") {
            const next = programArguments[i + 1];
            const parsed = parsePort(next);
            if (parsed)
                return parsed;
        }
        if (arg?.startsWith("--port=")) {
            const parsed = parsePort(arg.split("=", 2)[1]);
            if (parsed)
                return parsed;
        }
    }
    return null;
}
export function pickProbeHostForBind(bindMode, tailnetIPv4, customBindHost) {
    if (bindMode === "custom" && customBindHost?.trim()) {
        return customBindHost.trim();
    }
    if (bindMode === "tailnet")
        return tailnetIPv4 ?? "127.0.0.1";
    return "127.0.0.1";
}
const SAFE_DAEMON_ENV_KEYS = [
    "CLAWDBOT_PROFILE",
    "CLAWDBOT_STATE_DIR",
    "CLAWDBOT_CONFIG_PATH",
    "CLAWDBOT_GATEWAY_PORT",
    "CLAWDBOT_NIX_MODE",
];
export function filterDaemonEnv(env) {
    if (!env)
        return {};
    const filtered = {};
    for (const key of SAFE_DAEMON_ENV_KEYS) {
        const value = env[key];
        if (!value?.trim())
            continue;
        filtered[key] = value.trim();
    }
    return filtered;
}
export function safeDaemonEnv(env) {
    const filtered = filterDaemonEnv(env);
    return Object.entries(filtered).map(([key, value]) => `${key}=${value}`);
}
export function normalizeListenerAddress(raw) {
    let value = raw.trim();
    if (!value)
        return value;
    value = value.replace(/^TCP\s+/i, "");
    value = value.replace(/\s+\(LISTEN\)\s*$/i, "");
    return value.trim();
}
export function formatRuntimeStatus(runtime) {
    if (!runtime)
        return null;
    const status = runtime.status ?? "unknown";
    const details = [];
    if (runtime.pid)
        details.push(`pid ${runtime.pid}`);
    if (runtime.state && runtime.state.toLowerCase() !== status) {
        details.push(`state ${runtime.state}`);
    }
    if (runtime.subState)
        details.push(`sub ${runtime.subState}`);
    if (runtime.lastExitStatus !== undefined) {
        details.push(`last exit ${runtime.lastExitStatus}`);
    }
    if (runtime.lastExitReason)
        details.push(`reason ${runtime.lastExitReason}`);
    if (runtime.lastRunResult)
        details.push(`last run ${runtime.lastRunResult}`);
    if (runtime.lastRunTime)
        details.push(`last run time ${runtime.lastRunTime}`);
    if (runtime.detail)
        details.push(runtime.detail);
    return details.length > 0 ? `${status} (${details.join(", ")})` : status;
}
export function renderRuntimeHints(runtime, env = process.env) {
    if (!runtime)
        return [];
    const hints = [];
    const fileLog = (() => {
        try {
            return getResolvedLoggerSettings().file;
        }
        catch {
            return null;
        }
    })();
    if (runtime.missingUnit) {
        hints.push(`Service not installed. Run: ${formatCliCommand("clawdbot gateway install", env)}`);
        if (fileLog)
            hints.push(`File logs: ${fileLog}`);
        return hints;
    }
    if (runtime.status === "stopped") {
        if (fileLog)
            hints.push(`File logs: ${fileLog}`);
        if (process.platform === "darwin") {
            const logs = resolveGatewayLogPaths(env);
            hints.push(`Launchd stdout (if installed): ${logs.stdoutPath}`);
            hints.push(`Launchd stderr (if installed): ${logs.stderrPath}`);
        }
        else if (process.platform === "linux") {
            const unit = resolveGatewaySystemdServiceName(env.CLAWDBOT_PROFILE);
            hints.push(`Logs: journalctl --user -u ${unit}.service -n 200 --no-pager`);
        }
        else if (process.platform === "win32") {
            const task = resolveGatewayWindowsTaskName(env.CLAWDBOT_PROFILE);
            hints.push(`Logs: schtasks /Query /TN "${task}" /V /FO LIST`);
        }
    }
    return hints;
}
export function renderGatewayServiceStartHints(env = process.env) {
    const base = [
        formatCliCommand("clawdbot gateway install", env),
        formatCliCommand("clawdbot gateway", env),
    ];
    const profile = env.CLAWDBOT_PROFILE;
    switch (process.platform) {
        case "darwin": {
            const label = resolveGatewayLaunchAgentLabel(profile);
            return [...base, `launchctl bootstrap gui/$UID ~/Library/LaunchAgents/${label}.plist`];
        }
        case "linux": {
            const unit = resolveGatewaySystemdServiceName(profile);
            return [...base, `systemctl --user start ${unit}.service`];
        }
        case "win32": {
            const task = resolveGatewayWindowsTaskName(profile);
            return [...base, `schtasks /Run /TN "${task}"`];
        }
        default:
            return base;
    }
}
