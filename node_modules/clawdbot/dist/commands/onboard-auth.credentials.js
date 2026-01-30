import { resolveClawdbotAgentDir } from "../agents/agent-paths.js";
import { upsertAuthProfile } from "../agents/auth-profiles.js";
const resolveAuthAgentDir = (agentDir) => agentDir ?? resolveClawdbotAgentDir();
export async function writeOAuthCredentials(provider, creds, agentDir) {
    // Write to resolved agent dir so gateway finds credentials on startup.
    upsertAuthProfile({
        profileId: `${provider}:${creds.email ?? "default"}`,
        credential: {
            type: "oauth",
            provider,
            ...creds,
        },
        agentDir: resolveAuthAgentDir(agentDir),
    });
}
export async function setAnthropicApiKey(key, agentDir) {
    // Write to resolved agent dir so gateway finds credentials on startup.
    upsertAuthProfile({
        profileId: "anthropic:default",
        credential: {
            type: "api_key",
            provider: "anthropic",
            key,
        },
        agentDir: resolveAuthAgentDir(agentDir),
    });
}
export async function setGeminiApiKey(key, agentDir) {
    // Write to resolved agent dir so gateway finds credentials on startup.
    upsertAuthProfile({
        profileId: "google:default",
        credential: {
            type: "api_key",
            provider: "google",
            key,
        },
        agentDir: resolveAuthAgentDir(agentDir),
    });
}
export async function setMinimaxApiKey(key, agentDir) {
    // Write to resolved agent dir so gateway finds credentials on startup.
    upsertAuthProfile({
        profileId: "minimax:default",
        credential: {
            type: "api_key",
            provider: "minimax",
            key,
        },
        agentDir: resolveAuthAgentDir(agentDir),
    });
}
export async function setMoonshotApiKey(key, agentDir) {
    // Write to resolved agent dir so gateway finds credentials on startup.
    upsertAuthProfile({
        profileId: "moonshot:default",
        credential: {
            type: "api_key",
            provider: "moonshot",
            key,
        },
        agentDir: resolveAuthAgentDir(agentDir),
    });
}
export async function setKimiCodeApiKey(key, agentDir) {
    // Write to resolved agent dir so gateway finds credentials on startup.
    upsertAuthProfile({
        profileId: "kimi-code:default",
        credential: {
            type: "api_key",
            provider: "kimi-code",
            key,
        },
        agentDir: resolveAuthAgentDir(agentDir),
    });
}
export async function setSyntheticApiKey(key, agentDir) {
    // Write to resolved agent dir so gateway finds credentials on startup.
    upsertAuthProfile({
        profileId: "synthetic:default",
        credential: {
            type: "api_key",
            provider: "synthetic",
            key,
        },
        agentDir: resolveAuthAgentDir(agentDir),
    });
}
export async function setVeniceApiKey(key, agentDir) {
    // Write to resolved agent dir so gateway finds credentials on startup.
    upsertAuthProfile({
        profileId: "venice:default",
        credential: {
            type: "api_key",
            provider: "venice",
            key,
        },
        agentDir: resolveAuthAgentDir(agentDir),
    });
}
export const ZAI_DEFAULT_MODEL_REF = "zai/glm-4.7";
export const OPENROUTER_DEFAULT_MODEL_REF = "openrouter/auto";
export const VERCEL_AI_GATEWAY_DEFAULT_MODEL_REF = "vercel-ai-gateway/anthropic/claude-opus-4.5";
export async function setZaiApiKey(key, agentDir) {
    // Write to resolved agent dir so gateway finds credentials on startup.
    upsertAuthProfile({
        profileId: "zai:default",
        credential: {
            type: "api_key",
            provider: "zai",
            key,
        },
        agentDir: resolveAuthAgentDir(agentDir),
    });
}
export async function setOpenrouterApiKey(key, agentDir) {
    upsertAuthProfile({
        profileId: "openrouter:default",
        credential: {
            type: "api_key",
            provider: "openrouter",
            key,
        },
        agentDir: resolveAuthAgentDir(agentDir),
    });
}
export async function setVercelAiGatewayApiKey(key, agentDir) {
    upsertAuthProfile({
        profileId: "vercel-ai-gateway:default",
        credential: {
            type: "api_key",
            provider: "vercel-ai-gateway",
            key,
        },
        agentDir: resolveAuthAgentDir(agentDir),
    });
}
export async function setOpencodeZenApiKey(key, agentDir) {
    upsertAuthProfile({
        profileId: "opencode:default",
        credential: {
            type: "api_key",
            provider: "opencode",
            key,
        },
        agentDir: resolveAuthAgentDir(agentDir),
    });
}
