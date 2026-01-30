import { randomBytes } from "node:crypto";
import { createServer } from "node:http";
import { CHUTES_AUTHORIZE_ENDPOINT, exchangeChutesCodeForTokens, generateChutesPkce, parseOAuthCallbackInput, } from "../agents/chutes-oauth.js";
function buildAuthorizeUrl(params) {
    const qs = new URLSearchParams({
        client_id: params.clientId,
        redirect_uri: params.redirectUri,
        response_type: "code",
        scope: params.scopes.join(" "),
        state: params.state,
        code_challenge: params.challenge,
        code_challenge_method: "S256",
    });
    return `${CHUTES_AUTHORIZE_ENDPOINT}?${qs.toString()}`;
}
async function waitForLocalCallback(params) {
    const redirectUrl = new URL(params.redirectUri);
    if (redirectUrl.protocol !== "http:") {
        throw new Error(`Chutes OAuth redirect URI must be http:// (got ${params.redirectUri})`);
    }
    const hostname = redirectUrl.hostname || "127.0.0.1";
    const port = redirectUrl.port ? Number.parseInt(redirectUrl.port, 10) : 80;
    const expectedPath = redirectUrl.pathname || "/";
    return await new Promise((resolve, reject) => {
        let timeout = null;
        const server = createServer((req, res) => {
            try {
                const requestUrl = new URL(req.url ?? "/", redirectUrl.origin);
                if (requestUrl.pathname !== expectedPath) {
                    res.statusCode = 404;
                    res.setHeader("Content-Type", "text/plain; charset=utf-8");
                    res.end("Not found");
                    return;
                }
                const code = requestUrl.searchParams.get("code")?.trim();
                const state = requestUrl.searchParams.get("state")?.trim();
                if (!code) {
                    res.statusCode = 400;
                    res.setHeader("Content-Type", "text/plain; charset=utf-8");
                    res.end("Missing code");
                    return;
                }
                if (!state || state !== params.expectedState) {
                    res.statusCode = 400;
                    res.setHeader("Content-Type", "text/plain; charset=utf-8");
                    res.end("Invalid state");
                    return;
                }
                res.statusCode = 200;
                res.setHeader("Content-Type", "text/html; charset=utf-8");
                res.end([
                    "<!doctype html>",
                    "<html><head><meta charset='utf-8' /></head>",
                    "<body><h2>Chutes OAuth complete</h2>",
                    "<p>You can close this window and return to clawdbot.</p></body></html>",
                ].join(""));
                if (timeout)
                    clearTimeout(timeout);
                server.close();
                resolve({ code, state });
            }
            catch (err) {
                if (timeout)
                    clearTimeout(timeout);
                server.close();
                reject(err);
            }
        });
        server.once("error", (err) => {
            if (timeout)
                clearTimeout(timeout);
            server.close();
            reject(err);
        });
        server.listen(port, hostname, () => {
            params.onProgress?.(`Waiting for OAuth callback on ${redirectUrl.origin}${expectedPath}…`);
        });
        timeout = setTimeout(() => {
            try {
                server.close();
            }
            catch { }
            reject(new Error("OAuth callback timeout"));
        }, params.timeoutMs);
    });
}
export async function loginChutes(params) {
    const createPkce = params.createPkce ?? generateChutesPkce;
    const createState = params.createState ?? (() => randomBytes(16).toString("hex"));
    const { verifier, challenge } = createPkce();
    const state = createState();
    const timeoutMs = params.timeoutMs ?? 3 * 60 * 1000;
    const url = buildAuthorizeUrl({
        clientId: params.app.clientId,
        redirectUri: params.app.redirectUri,
        scopes: params.app.scopes,
        state,
        challenge,
    });
    let codeAndState;
    if (params.manual) {
        await params.onAuth({ url });
        params.onProgress?.("Waiting for redirect URL…");
        const input = await params.onPrompt({
            message: "Paste the redirect URL (or authorization code)",
            placeholder: `${params.app.redirectUri}?code=...&state=...`,
        });
        const parsed = parseOAuthCallbackInput(String(input), state);
        if ("error" in parsed)
            throw new Error(parsed.error);
        if (parsed.state !== state)
            throw new Error("Invalid OAuth state");
        codeAndState = parsed;
    }
    else {
        const callback = waitForLocalCallback({
            redirectUri: params.app.redirectUri,
            expectedState: state,
            timeoutMs,
            onProgress: params.onProgress,
        }).catch(async () => {
            params.onProgress?.("OAuth callback not detected; paste redirect URL…");
            const input = await params.onPrompt({
                message: "Paste the redirect URL (or authorization code)",
                placeholder: `${params.app.redirectUri}?code=...&state=...`,
            });
            const parsed = parseOAuthCallbackInput(String(input), state);
            if ("error" in parsed)
                throw new Error(parsed.error);
            if (parsed.state !== state)
                throw new Error("Invalid OAuth state");
            return parsed;
        });
        await params.onAuth({ url });
        codeAndState = await callback;
    }
    params.onProgress?.("Exchanging code for tokens…");
    return await exchangeChutesCodeForTokens({
        app: params.app,
        code: codeAndState.code,
        codeVerifier: verifier,
        fetchFn: params.fetchFn,
    });
}
