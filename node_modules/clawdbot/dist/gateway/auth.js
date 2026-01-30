import { timingSafeEqual } from "node:crypto";
import { isTrustedProxyAddress, resolveGatewayClientIp } from "./net.js";
function safeEqual(a, b) {
    if (a.length !== b.length)
        return false;
    return timingSafeEqual(Buffer.from(a), Buffer.from(b));
}
function isLoopbackAddress(ip) {
    if (!ip)
        return false;
    if (ip === "127.0.0.1")
        return true;
    if (ip.startsWith("127."))
        return true;
    if (ip === "::1")
        return true;
    if (ip.startsWith("::ffff:127."))
        return true;
    return false;
}
function getHostName(hostHeader) {
    const host = (hostHeader ?? "").trim().toLowerCase();
    if (!host)
        return "";
    if (host.startsWith("[")) {
        const end = host.indexOf("]");
        if (end !== -1)
            return host.slice(1, end);
    }
    const [name] = host.split(":");
    return name ?? "";
}
function headerValue(value) {
    return Array.isArray(value) ? value[0] : value;
}
function resolveRequestClientIp(req, trustedProxies) {
    if (!req)
        return undefined;
    return resolveGatewayClientIp({
        remoteAddr: req.socket?.remoteAddress ?? "",
        forwardedFor: headerValue(req.headers?.["x-forwarded-for"]),
        realIp: headerValue(req.headers?.["x-real-ip"]),
        trustedProxies,
    });
}
function isLocalDirectRequest(req, trustedProxies) {
    if (!req)
        return false;
    const clientIp = resolveRequestClientIp(req, trustedProxies) ?? "";
    if (!isLoopbackAddress(clientIp))
        return false;
    const host = getHostName(req.headers?.host);
    const hostIsLocal = host === "localhost" || host === "127.0.0.1" || host === "::1";
    const hostIsTailscaleServe = host.endsWith(".ts.net");
    const hasForwarded = Boolean(req.headers?.["x-forwarded-for"] ||
        req.headers?.["x-real-ip"] ||
        req.headers?.["x-forwarded-host"]);
    const remoteIsTrustedProxy = isTrustedProxyAddress(req.socket?.remoteAddress, trustedProxies);
    return (hostIsLocal || hostIsTailscaleServe) && (!hasForwarded || remoteIsTrustedProxy);
}
function getTailscaleUser(req) {
    if (!req)
        return null;
    const login = req.headers["tailscale-user-login"];
    if (typeof login !== "string" || !login.trim())
        return null;
    const nameRaw = req.headers["tailscale-user-name"];
    const profilePic = req.headers["tailscale-user-profile-pic"];
    const name = typeof nameRaw === "string" && nameRaw.trim() ? nameRaw.trim() : login.trim();
    return {
        login: login.trim(),
        name,
        profilePic: typeof profilePic === "string" && profilePic.trim() ? profilePic.trim() : undefined,
    };
}
function hasTailscaleProxyHeaders(req) {
    if (!req)
        return false;
    return Boolean(req.headers["x-forwarded-for"] &&
        req.headers["x-forwarded-proto"] &&
        req.headers["x-forwarded-host"]);
}
function isTailscaleProxyRequest(req) {
    if (!req)
        return false;
    return isLoopbackAddress(req.socket?.remoteAddress) && hasTailscaleProxyHeaders(req);
}
export function resolveGatewayAuth(params) {
    const authConfig = params.authConfig ?? {};
    const env = params.env ?? process.env;
    const token = authConfig.token ?? env.CLAWDBOT_GATEWAY_TOKEN ?? undefined;
    const password = authConfig.password ?? env.CLAWDBOT_GATEWAY_PASSWORD ?? undefined;
    const mode = authConfig.mode ?? (password ? "password" : token ? "token" : "none");
    const allowTailscale = authConfig.allowTailscale ?? (params.tailscaleMode === "serve" && mode !== "password");
    return {
        mode,
        token,
        password,
        allowTailscale,
    };
}
export function assertGatewayAuthConfigured(auth) {
    if (auth.mode === "token" && !auth.token) {
        throw new Error("gateway auth mode is token, but no token was configured (set gateway.auth.token or CLAWDBOT_GATEWAY_TOKEN)");
    }
    if (auth.mode === "password" && !auth.password) {
        throw new Error("gateway auth mode is password, but no password was configured");
    }
}
export async function authorizeGatewayConnect(params) {
    const { auth, connectAuth, req, trustedProxies } = params;
    const localDirect = isLocalDirectRequest(req, trustedProxies);
    if (auth.allowTailscale && !localDirect) {
        const tailscaleUser = getTailscaleUser(req);
        const tailscaleProxy = isTailscaleProxyRequest(req);
        if (tailscaleUser && tailscaleProxy) {
            return {
                ok: true,
                method: "tailscale",
                user: tailscaleUser.login,
            };
        }
        if (auth.mode === "none") {
            if (!tailscaleUser) {
                return { ok: false, reason: "tailscale_user_missing" };
            }
            if (!tailscaleProxy) {
                return { ok: false, reason: "tailscale_proxy_missing" };
            }
        }
    }
    if (auth.mode === "none") {
        return { ok: true, method: "none" };
    }
    if (auth.mode === "token") {
        if (!auth.token) {
            return { ok: false, reason: "token_missing_config" };
        }
        if (!connectAuth?.token) {
            return { ok: false, reason: "token_missing" };
        }
        if (connectAuth.token !== auth.token) {
            return { ok: false, reason: "token_mismatch" };
        }
        return { ok: true, method: "token" };
    }
    if (auth.mode === "password") {
        const password = connectAuth?.password;
        if (!auth.password) {
            return { ok: false, reason: "password_missing_config" };
        }
        if (!password) {
            return { ok: false, reason: "password_missing" };
        }
        if (!safeEqual(password, auth.password)) {
            return { ok: false, reason: "password_mismatch" };
        }
        return { ok: true, method: "password" };
    }
    return { ok: false, reason: "unauthorized" };
}
