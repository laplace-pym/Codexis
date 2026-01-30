import { intro, note, outro, spinner } from "@clack/prompts";
import { ensureAuthProfileStore, upsertAuthProfile } from "../agents/auth-profiles.js";
import { updateConfig } from "../commands/models/shared.js";
import { applyAuthProfileConfig } from "../commands/onboard-auth.js";
import { logConfigUpdated } from "../config/logging.js";
import { stylePromptTitle } from "../terminal/prompt-style.js";
const CLIENT_ID = "Iv1.b507a08c87ecfe98";
const DEVICE_CODE_URL = "https://github.com/login/device/code";
const ACCESS_TOKEN_URL = "https://github.com/login/oauth/access_token";
function parseJsonResponse(value) {
    if (!value || typeof value !== "object") {
        throw new Error("Unexpected response from GitHub");
    }
    return value;
}
async function requestDeviceCode(params) {
    const body = new URLSearchParams({
        client_id: CLIENT_ID,
        scope: params.scope,
    });
    const res = await fetch(DEVICE_CODE_URL, {
        method: "POST",
        headers: {
            Accept: "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        body,
    });
    if (!res.ok) {
        throw new Error(`GitHub device code failed: HTTP ${res.status}`);
    }
    const json = parseJsonResponse(await res.json());
    if (!json.device_code || !json.user_code || !json.verification_uri) {
        throw new Error("GitHub device code response missing fields");
    }
    return json;
}
async function pollForAccessToken(params) {
    const bodyBase = new URLSearchParams({
        client_id: CLIENT_ID,
        device_code: params.deviceCode,
        grant_type: "urn:ietf:params:oauth:grant-type:device_code",
    });
    while (Date.now() < params.expiresAt) {
        const res = await fetch(ACCESS_TOKEN_URL, {
            method: "POST",
            headers: {
                Accept: "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            body: bodyBase,
        });
        if (!res.ok) {
            throw new Error(`GitHub device token failed: HTTP ${res.status}`);
        }
        const json = parseJsonResponse(await res.json());
        if ("access_token" in json && typeof json.access_token === "string") {
            return json.access_token;
        }
        const err = "error" in json ? json.error : "unknown";
        if (err === "authorization_pending") {
            await new Promise((r) => setTimeout(r, params.intervalMs));
            continue;
        }
        if (err === "slow_down") {
            await new Promise((r) => setTimeout(r, params.intervalMs + 2000));
            continue;
        }
        if (err === "expired_token") {
            throw new Error("GitHub device code expired; run login again");
        }
        if (err === "access_denied") {
            throw new Error("GitHub login cancelled");
        }
        throw new Error(`GitHub device flow error: ${err}`);
    }
    throw new Error("GitHub device code expired; run login again");
}
export async function githubCopilotLoginCommand(opts, runtime) {
    if (!process.stdin.isTTY) {
        throw new Error("github-copilot login requires an interactive TTY.");
    }
    intro(stylePromptTitle("GitHub Copilot login"));
    const profileId = opts.profileId?.trim() || "github-copilot:github";
    const store = ensureAuthProfileStore(undefined, {
        allowKeychainPrompt: false,
    });
    if (store.profiles[profileId] && !opts.yes) {
        note(`Auth profile already exists: ${profileId}\nRe-running will overwrite it.`, stylePromptTitle("Existing credentials"));
    }
    const spin = spinner();
    spin.start("Requesting device code from GitHub...");
    const device = await requestDeviceCode({ scope: "read:user" });
    spin.stop("Device code ready");
    note([`Visit: ${device.verification_uri}`, `Code: ${device.user_code}`].join("\n"), stylePromptTitle("Authorize"));
    const expiresAt = Date.now() + device.expires_in * 1000;
    const intervalMs = Math.max(1000, device.interval * 1000);
    const polling = spinner();
    polling.start("Waiting for GitHub authorization...");
    const accessToken = await pollForAccessToken({
        deviceCode: device.device_code,
        intervalMs,
        expiresAt,
    });
    polling.stop("GitHub access token acquired");
    upsertAuthProfile({
        profileId,
        credential: {
            type: "token",
            provider: "github-copilot",
            token: accessToken,
            // GitHub device flow token doesn't reliably include expiry here.
            // Leave expires unset; we'll exchange into Copilot token plus expiry later.
        },
    });
    await updateConfig((cfg) => applyAuthProfileConfig(cfg, {
        provider: "github-copilot",
        profileId,
        mode: "token",
    }));
    logConfigUpdated(runtime);
    runtime.log(`Auth profile: ${profileId} (github-copilot/token)`);
    outro("Done");
}
