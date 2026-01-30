import { resolveDefaultAgentId } from "../../agents/agent-scope.js";
import { getHealthSnapshot } from "../../commands/health.js";
import { CONFIG_PATH_CLAWDBOT, STATE_DIR_CLAWDBOT, loadConfig } from "../../config/config.js";
import { resolveMainSessionKey } from "../../config/sessions.js";
import { normalizeMainKey } from "../../routing/session-key.js";
import { listSystemPresence } from "../../infra/system-presence.js";
let presenceVersion = 1;
let healthVersion = 1;
let healthCache = null;
let healthRefresh = null;
let broadcastHealthUpdate = null;
export function buildGatewaySnapshot() {
    const cfg = loadConfig();
    const defaultAgentId = resolveDefaultAgentId(cfg);
    const mainKey = normalizeMainKey(cfg.session?.mainKey);
    const mainSessionKey = resolveMainSessionKey(cfg);
    const scope = cfg.session?.scope ?? "per-sender";
    const presence = listSystemPresence();
    const uptimeMs = Math.round(process.uptime() * 1000);
    // Health is async; caller should await getHealthSnapshot and replace later if needed.
    const emptyHealth = {};
    return {
        presence,
        health: emptyHealth,
        stateVersion: { presence: presenceVersion, health: healthVersion },
        uptimeMs,
        // Surface resolved paths so UIs can display the true config location.
        configPath: CONFIG_PATH_CLAWDBOT,
        stateDir: STATE_DIR_CLAWDBOT,
        sessionDefaults: {
            defaultAgentId,
            mainKey,
            mainSessionKey,
            scope,
        },
    };
}
export function getHealthCache() {
    return healthCache;
}
export function getHealthVersion() {
    return healthVersion;
}
export function incrementPresenceVersion() {
    presenceVersion += 1;
    return presenceVersion;
}
export function getPresenceVersion() {
    return presenceVersion;
}
export function setBroadcastHealthUpdate(fn) {
    broadcastHealthUpdate = fn;
}
export async function refreshGatewayHealthSnapshot(opts) {
    if (!healthRefresh) {
        healthRefresh = (async () => {
            const snap = await getHealthSnapshot({ probe: opts?.probe });
            healthCache = snap;
            healthVersion += 1;
            if (broadcastHealthUpdate) {
                broadcastHealthUpdate(snap);
            }
            return snap;
        })().finally(() => {
            healthRefresh = null;
        });
    }
    return healthRefresh;
}
