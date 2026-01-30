import os from "node:os";
import { logDebug, logWarn } from "../logger.js";
import { getLogger } from "../logging.js";
import { ignoreCiaoCancellationRejection } from "./bonjour-ciao.js";
import { formatBonjourError } from "./bonjour-errors.js";
import { isTruthyEnvValue } from "./env.js";
import { registerUnhandledRejectionHandler } from "./unhandled-rejections.js";
function isDisabledByEnv() {
    if (isTruthyEnvValue(process.env.CLAWDBOT_DISABLE_BONJOUR))
        return true;
    if (process.env.NODE_ENV === "test")
        return true;
    if (process.env.VITEST)
        return true;
    return false;
}
function safeServiceName(name) {
    const trimmed = name.trim();
    return trimmed.length > 0 ? trimmed : "Clawdbot";
}
function prettifyInstanceName(name) {
    const normalized = name.trim().replace(/\s+/g, " ");
    return normalized.replace(/\s+\(Clawdbot\)\s*$/i, "").trim() || normalized;
}
function serviceSummary(label, svc) {
    let fqdn = "unknown";
    let hostname = "unknown";
    let port = -1;
    try {
        fqdn = svc.getFQDN();
    }
    catch {
        // ignore
    }
    try {
        hostname = svc.getHostname();
    }
    catch {
        // ignore
    }
    try {
        port = svc.getPort();
    }
    catch {
        // ignore
    }
    const state = typeof svc.serviceState === "string" ? svc.serviceState : "unknown";
    return `${label} fqdn=${fqdn} host=${hostname} port=${port} state=${state}`;
}
export async function startGatewayBonjourAdvertiser(opts) {
    if (isDisabledByEnv()) {
        return { stop: async () => { } };
    }
    const { getResponder, Protocol } = await import("@homebridge/ciao");
    const responder = getResponder();
    // mDNS service instance names are single DNS labels; dots in hostnames (like
    // `Mac.localdomain`) can confuse some resolvers/browsers and break discovery.
    // Keep only the first label and normalize away a trailing `.local`.
    const hostname = os
        .hostname()
        .replace(/\.local$/i, "")
        .split(".")[0]
        .trim() || "clawdbot";
    const instanceName = typeof opts.instanceName === "string" && opts.instanceName.trim()
        ? opts.instanceName.trim()
        : `${hostname} (Clawdbot)`;
    const displayName = prettifyInstanceName(instanceName);
    const txtBase = {
        role: "gateway",
        gatewayPort: String(opts.gatewayPort),
        lanHost: `${hostname}.local`,
        displayName,
    };
    if (opts.gatewayTlsEnabled) {
        txtBase.gatewayTls = "1";
        if (opts.gatewayTlsFingerprintSha256) {
            txtBase.gatewayTlsSha256 = opts.gatewayTlsFingerprintSha256;
        }
    }
    if (typeof opts.canvasPort === "number" && opts.canvasPort > 0) {
        txtBase.canvasPort = String(opts.canvasPort);
    }
    if (typeof opts.tailnetDns === "string" && opts.tailnetDns.trim()) {
        txtBase.tailnetDns = opts.tailnetDns.trim();
    }
    if (typeof opts.cliPath === "string" && opts.cliPath.trim()) {
        txtBase.cliPath = opts.cliPath.trim();
    }
    const services = [];
    const gateway = responder.createService({
        name: safeServiceName(instanceName),
        type: "clawdbot-gw",
        protocol: "tcp" /* Protocol.TCP */,
        port: opts.gatewayPort,
        domain: "local",
        hostname,
        txt: {
            ...txtBase,
            sshPort: String(opts.sshPort ?? 22),
            transport: "gateway",
        },
    });
    services.push({
        label: "gateway",
        svc: gateway,
    });
    let ciaoCancellationRejectionHandler;
    if (services.length > 0) {
        ciaoCancellationRejectionHandler = registerUnhandledRejectionHandler(ignoreCiaoCancellationRejection);
    }
    logDebug(`bonjour: starting (hostname=${hostname}, instance=${JSON.stringify(safeServiceName(instanceName))}, gatewayPort=${opts.gatewayPort}, sshPort=${opts.sshPort ?? 22})`);
    for (const { label, svc } of services) {
        try {
            svc.on("name-change", (name) => {
                const next = typeof name === "string" ? name : String(name);
                logWarn(`bonjour: ${label} name conflict resolved; newName=${JSON.stringify(next)}`);
            });
            svc.on("hostname-change", (nextHostname) => {
                const next = typeof nextHostname === "string" ? nextHostname : String(nextHostname);
                logWarn(`bonjour: ${label} hostname conflict resolved; newHostname=${JSON.stringify(next)}`);
            });
        }
        catch (err) {
            logDebug(`bonjour: failed to attach listeners for ${label}: ${String(err)}`);
        }
    }
    // Do not block gateway startup on mDNS probing/announce. Advertising can take
    // multiple seconds depending on network state; the gateway should come up even
    // if Bonjour is slow or fails.
    for (const { label, svc } of services) {
        try {
            void svc
                .advertise()
                .then(() => {
                // Keep this out of stdout/stderr (menubar + tests) but capture in the rolling log.
                getLogger().info(`bonjour: advertised ${serviceSummary(label, svc)}`);
            })
                .catch((err) => {
                logWarn(`bonjour: advertise failed (${serviceSummary(label, svc)}): ${formatBonjourError(err)}`);
            });
        }
        catch (err) {
            logWarn(`bonjour: advertise threw (${serviceSummary(label, svc)}): ${formatBonjourError(err)}`);
        }
    }
    // Watchdog: if we ever end up in an unannounced state (e.g. after sleep/wake or
    // interface churn), try to re-advertise instead of requiring a full gateway restart.
    const lastRepairAttempt = new Map();
    const watchdog = setInterval(() => {
        for (const { label, svc } of services) {
            const stateUnknown = svc.serviceState;
            if (typeof stateUnknown !== "string")
                continue;
            if (stateUnknown === "announced" || stateUnknown === "announcing")
                continue;
            let key = label;
            try {
                key = `${label}:${svc.getFQDN()}`;
            }
            catch {
                // ignore
            }
            const now = Date.now();
            const last = lastRepairAttempt.get(key) ?? 0;
            if (now - last < 30_000)
                continue;
            lastRepairAttempt.set(key, now);
            logWarn(`bonjour: watchdog detected non-announced service; attempting re-advertise (${serviceSummary(label, svc)})`);
            try {
                void svc.advertise().catch((err) => {
                    logWarn(`bonjour: watchdog advertise failed (${serviceSummary(label, svc)}): ${formatBonjourError(err)}`);
                });
            }
            catch (err) {
                logWarn(`bonjour: watchdog advertise threw (${serviceSummary(label, svc)}): ${formatBonjourError(err)}`);
            }
        }
    }, 60_000);
    watchdog.unref?.();
    return {
        stop: async () => {
            clearInterval(watchdog);
            for (const { svc } of services) {
                try {
                    await svc.destroy();
                }
                catch {
                    /* ignore */
                }
            }
            try {
                await responder.shutdown();
            }
            catch {
                /* ignore */
            }
            finally {
                ciaoCancellationRejectionHandler?.();
            }
        },
    };
}
