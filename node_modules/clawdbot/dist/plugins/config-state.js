import { defaultSlotIdForKey } from "./slots.js";
export const BUNDLED_ENABLED_BY_DEFAULT = new Set();
const normalizeList = (value) => {
    if (!Array.isArray(value))
        return [];
    return value.map((entry) => (typeof entry === "string" ? entry.trim() : "")).filter(Boolean);
};
const normalizeSlotValue = (value) => {
    if (typeof value !== "string")
        return undefined;
    const trimmed = value.trim();
    if (!trimmed)
        return undefined;
    if (trimmed.toLowerCase() === "none")
        return null;
    return trimmed;
};
const normalizePluginEntries = (entries) => {
    if (!entries || typeof entries !== "object" || Array.isArray(entries)) {
        return {};
    }
    const normalized = {};
    for (const [key, value] of Object.entries(entries)) {
        if (!key.trim())
            continue;
        if (!value || typeof value !== "object" || Array.isArray(value)) {
            normalized[key] = {};
            continue;
        }
        const entry = value;
        normalized[key] = {
            enabled: typeof entry.enabled === "boolean" ? entry.enabled : undefined,
            config: "config" in entry ? entry.config : undefined,
        };
    }
    return normalized;
};
export const normalizePluginsConfig = (config) => {
    const memorySlot = normalizeSlotValue(config?.slots?.memory);
    return {
        enabled: config?.enabled !== false,
        allow: normalizeList(config?.allow),
        deny: normalizeList(config?.deny),
        loadPaths: normalizeList(config?.load?.paths),
        slots: {
            memory: memorySlot === undefined ? defaultSlotIdForKey("memory") : memorySlot,
        },
        entries: normalizePluginEntries(config?.entries),
    };
};
export function resolveEnableState(id, origin, config) {
    if (!config.enabled) {
        return { enabled: false, reason: "plugins disabled" };
    }
    if (config.deny.includes(id)) {
        return { enabled: false, reason: "blocked by denylist" };
    }
    if (config.allow.length > 0 && !config.allow.includes(id)) {
        return { enabled: false, reason: "not in allowlist" };
    }
    if (config.slots.memory === id) {
        return { enabled: true };
    }
    const entry = config.entries[id];
    if (entry?.enabled === true) {
        return { enabled: true };
    }
    if (entry?.enabled === false) {
        return { enabled: false, reason: "disabled in config" };
    }
    if (origin === "bundled" && BUNDLED_ENABLED_BY_DEFAULT.has(id)) {
        return { enabled: true };
    }
    if (origin === "bundled") {
        return { enabled: false, reason: "bundled (disabled by default)" };
    }
    return { enabled: true };
}
export function resolveMemorySlotDecision(params) {
    if (params.kind !== "memory")
        return { enabled: true };
    if (params.slot === null) {
        return { enabled: false, reason: "memory slot disabled" };
    }
    if (typeof params.slot === "string") {
        if (params.slot === params.id) {
            return { enabled: true, selected: true };
        }
        return {
            enabled: false,
            reason: `memory slot set to "${params.slot}"`,
        };
    }
    if (params.selectedId && params.selectedId !== params.id) {
        return {
            enabled: false,
            reason: `memory slot already filled by "${params.selectedId}"`,
        };
    }
    return { enabled: true, selected: true };
}
