import { resolveGatewayPort } from "../../config/config.js";
import { pickPrimaryTailnetIPv4 } from "../../infra/tailnet.js";
import { colorize, theme } from "../../terminal/theme.js";
function parseIntOrNull(value) {
    const s = typeof value === "string"
        ? value.trim()
        : typeof value === "number" || typeof value === "bigint"
            ? String(value)
            : "";
    if (!s)
        return null;
    const n = Number.parseInt(s, 10);
    return Number.isFinite(n) ? n : null;
}
export function parseTimeoutMs(raw, fallbackMs) {
    const value = typeof raw === "string"
        ? raw.trim()
        : typeof raw === "number" || typeof raw === "bigint"
            ? String(raw)
            : "";
    if (!value)
        return fallbackMs;
    const parsed = Number.parseInt(value, 10);
    if (!Number.isFinite(parsed) || parsed <= 0) {
        throw new Error(`invalid --timeout: ${value}`);
    }
    return parsed;
}
function normalizeWsUrl(value) {
    const trimmed = value.trim();
    if (!trimmed)
        return null;
    if (!trimmed.startsWith("ws://") && !trimmed.startsWith("wss://"))
        return null;
    return trimmed;
}
export function resolveTargets(cfg, explicitUrl) {
    const targets = [];
    const add = (t) => {
        if (!targets.some((x) => x.url === t.url))
            targets.push(t);
    };
    const explicit = typeof explicitUrl === "string" ? normalizeWsUrl(explicitUrl) : null;
    if (explicit)
        add({ id: "explicit", kind: "explicit", url: explicit, active: true });
    const remoteUrl = typeof cfg.gateway?.remote?.url === "string" ? normalizeWsUrl(cfg.gateway.remote.url) : null;
    if (remoteUrl) {
        add({
            id: "configRemote",
            kind: "configRemote",
            url: remoteUrl,
            active: cfg.gateway?.mode === "remote",
        });
    }
    const port = resolveGatewayPort(cfg);
    add({
        id: "localLoopback",
        kind: "localLoopback",
        url: `ws://127.0.0.1:${port}`,
        active: cfg.gateway?.mode !== "remote",
    });
    return targets;
}
export function resolveProbeBudgetMs(overallMs, kind) {
    if (kind === "localLoopback")
        return Math.min(800, overallMs);
    if (kind === "sshTunnel")
        return Math.min(2000, overallMs);
    return Math.min(1500, overallMs);
}
export function sanitizeSshTarget(value) {
    if (typeof value !== "string")
        return null;
    const trimmed = value.trim();
    if (!trimmed)
        return null;
    return trimmed.replace(/^ssh\\s+/, "");
}
export function resolveAuthForTarget(cfg, target, overrides) {
    const tokenOverride = overrides.token?.trim() ? overrides.token.trim() : undefined;
    const passwordOverride = overrides.password?.trim() ? overrides.password.trim() : undefined;
    if (tokenOverride || passwordOverride) {
        return { token: tokenOverride, password: passwordOverride };
    }
    if (target.kind === "configRemote" || target.kind === "sshTunnel") {
        const token = typeof cfg.gateway?.remote?.token === "string" ? cfg.gateway.remote.token.trim() : "";
        const remotePassword = cfg.gateway?.remote?.password;
        const password = typeof remotePassword === "string" ? remotePassword.trim() : "";
        return {
            token: token.length > 0 ? token : undefined,
            password: password.length > 0 ? password : undefined,
        };
    }
    const envToken = process.env.CLAWDBOT_GATEWAY_TOKEN?.trim() || "";
    const envPassword = process.env.CLAWDBOT_GATEWAY_PASSWORD?.trim() || "";
    const cfgToken = typeof cfg.gateway?.auth?.token === "string" ? cfg.gateway.auth.token.trim() : "";
    const cfgPassword = typeof cfg.gateway?.auth?.password === "string" ? cfg.gateway.auth.password.trim() : "";
    return {
        token: envToken || cfgToken || undefined,
        password: envPassword || cfgPassword || undefined,
    };
}
export function pickGatewaySelfPresence(presence) {
    if (!Array.isArray(presence))
        return null;
    const entries = presence;
    const self = entries.find((e) => e.mode === "gateway" && e.reason === "self") ??
        entries.find((e) => typeof e.text === "string" && String(e.text).startsWith("Gateway:")) ??
        null;
    if (!self)
        return null;
    return {
        host: typeof self.host === "string" ? self.host : undefined,
        ip: typeof self.ip === "string" ? self.ip : undefined,
        version: typeof self.version === "string" ? self.version : undefined,
        platform: typeof self.platform === "string" ? self.platform : undefined,
    };
}
export function extractConfigSummary(snapshotUnknown) {
    const snap = snapshotUnknown;
    const path = typeof snap?.path === "string" ? snap.path : null;
    const exists = Boolean(snap?.exists);
    const valid = Boolean(snap?.valid);
    const issuesRaw = Array.isArray(snap?.issues) ? snap.issues : [];
    const legacyRaw = Array.isArray(snap?.legacyIssues) ? snap.legacyIssues : [];
    const cfg = (snap?.config ?? {});
    const gateway = (cfg.gateway ?? {});
    const discovery = (cfg.discovery ?? {});
    const wideArea = (discovery.wideArea ?? {});
    const remote = (gateway.remote ?? {});
    const auth = (gateway.auth ?? {});
    const controlUi = (gateway.controlUi ?? {});
    const tailscale = (gateway.tailscale ?? {});
    const authMode = typeof auth.mode === "string" ? auth.mode : null;
    const authTokenConfigured = typeof auth.token === "string" ? auth.token.trim().length > 0 : false;
    const authPasswordConfigured = typeof auth.password === "string" ? auth.password.trim().length > 0 : false;
    const remoteUrl = typeof remote.url === "string" ? normalizeWsUrl(remote.url) : null;
    const remoteTokenConfigured = typeof remote.token === "string" ? remote.token.trim().length > 0 : false;
    const remotePasswordConfigured = typeof remote.password === "string" ? String(remote.password).trim().length > 0 : false;
    const wideAreaEnabled = typeof wideArea.enabled === "boolean" ? wideArea.enabled : null;
    return {
        path,
        exists,
        valid,
        issues: issuesRaw
            .filter((i) => Boolean(i && typeof i.path === "string" && typeof i.message === "string"))
            .map((i) => ({ path: i.path, message: i.message })),
        legacyIssues: legacyRaw
            .filter((i) => Boolean(i && typeof i.path === "string" && typeof i.message === "string"))
            .map((i) => ({ path: i.path, message: i.message })),
        gateway: {
            mode: typeof gateway.mode === "string" ? gateway.mode : null,
            bind: typeof gateway.bind === "string" ? gateway.bind : null,
            port: parseIntOrNull(gateway.port),
            controlUiEnabled: typeof controlUi.enabled === "boolean" ? controlUi.enabled : null,
            controlUiBasePath: typeof controlUi.basePath === "string" ? controlUi.basePath : null,
            authMode,
            authTokenConfigured,
            authPasswordConfigured,
            remoteUrl,
            remoteTokenConfigured,
            remotePasswordConfigured,
            tailscaleMode: typeof tailscale.mode === "string" ? tailscale.mode : null,
        },
        discovery: { wideAreaEnabled },
    };
}
export function buildNetworkHints(cfg) {
    const tailnetIPv4 = pickPrimaryTailnetIPv4();
    const port = resolveGatewayPort(cfg);
    return {
        localLoopbackUrl: `ws://127.0.0.1:${port}`,
        localTailnetUrl: tailnetIPv4 ? `ws://${tailnetIPv4}:${port}` : null,
        tailnetIPv4: tailnetIPv4 ?? null,
    };
}
export function renderTargetHeader(target, rich) {
    const kindLabel = target.kind === "localLoopback"
        ? "Local loopback"
        : target.kind === "sshTunnel"
            ? "Remote over SSH"
            : target.kind === "configRemote"
                ? target.active
                    ? "Remote (configured)"
                    : "Remote (configured, inactive)"
                : "URL (explicit)";
    return `${colorize(rich, theme.heading, kindLabel)} ${colorize(rich, theme.muted, target.url)}`;
}
export function renderProbeSummaryLine(probe, rich) {
    if (probe.ok) {
        const latency = typeof probe.connectLatencyMs === "number" ? `${probe.connectLatencyMs}ms` : "unknown";
        return `${colorize(rich, theme.success, "Connect: ok")} (${latency}) · ${colorize(rich, theme.success, "RPC: ok")}`;
    }
    const detail = probe.error ? ` - ${probe.error}` : "";
    if (probe.connectLatencyMs != null) {
        const latency = typeof probe.connectLatencyMs === "number" ? `${probe.connectLatencyMs}ms` : "unknown";
        return `${colorize(rich, theme.success, "Connect: ok")} (${latency}) · ${colorize(rich, theme.error, "RPC: failed")}${detail}`;
    }
    return `${colorize(rich, theme.error, "Connect: failed")}${detail}`;
}
