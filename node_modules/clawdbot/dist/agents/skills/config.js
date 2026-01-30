import fs from "node:fs";
import path from "node:path";
import { resolveSkillKey } from "./frontmatter.js";
const DEFAULT_CONFIG_VALUES = {
    "browser.enabled": true,
};
function isTruthy(value) {
    if (value === undefined || value === null)
        return false;
    if (typeof value === "boolean")
        return value;
    if (typeof value === "number")
        return value !== 0;
    if (typeof value === "string")
        return value.trim().length > 0;
    return true;
}
export function resolveConfigPath(config, pathStr) {
    const parts = pathStr.split(".").filter(Boolean);
    let current = config;
    for (const part of parts) {
        if (typeof current !== "object" || current === null)
            return undefined;
        current = current[part];
    }
    return current;
}
export function isConfigPathTruthy(config, pathStr) {
    const value = resolveConfigPath(config, pathStr);
    if (value === undefined && pathStr in DEFAULT_CONFIG_VALUES) {
        return DEFAULT_CONFIG_VALUES[pathStr] === true;
    }
    return isTruthy(value);
}
export function resolveSkillConfig(config, skillKey) {
    const skills = config?.skills?.entries;
    if (!skills || typeof skills !== "object")
        return undefined;
    const entry = skills[skillKey];
    if (!entry || typeof entry !== "object")
        return undefined;
    return entry;
}
export function resolveRuntimePlatform() {
    return process.platform;
}
function normalizeAllowlist(input) {
    if (!input)
        return undefined;
    if (!Array.isArray(input))
        return undefined;
    const normalized = input.map((entry) => String(entry).trim()).filter(Boolean);
    return normalized.length > 0 ? normalized : undefined;
}
function isBundledSkill(entry) {
    return entry.skill.source === "clawdbot-bundled";
}
export function resolveBundledAllowlist(config) {
    return normalizeAllowlist(config?.skills?.allowBundled);
}
export function isBundledSkillAllowed(entry, allowlist) {
    if (!allowlist || allowlist.length === 0)
        return true;
    if (!isBundledSkill(entry))
        return true;
    const key = resolveSkillKey(entry.skill, entry);
    return allowlist.includes(key) || allowlist.includes(entry.skill.name);
}
export function hasBinary(bin) {
    const pathEnv = process.env.PATH ?? "";
    const parts = pathEnv.split(path.delimiter).filter(Boolean);
    for (const part of parts) {
        const candidate = path.join(part, bin);
        try {
            fs.accessSync(candidate, fs.constants.X_OK);
            return true;
        }
        catch {
            // keep scanning
        }
    }
    return false;
}
export function shouldIncludeSkill(params) {
    const { entry, config, eligibility } = params;
    const skillKey = resolveSkillKey(entry.skill, entry);
    const skillConfig = resolveSkillConfig(config, skillKey);
    const allowBundled = normalizeAllowlist(config?.skills?.allowBundled);
    const osList = entry.clawdbot?.os ?? [];
    const remotePlatforms = eligibility?.remote?.platforms ?? [];
    if (skillConfig?.enabled === false)
        return false;
    if (!isBundledSkillAllowed(entry, allowBundled))
        return false;
    if (osList.length > 0 &&
        !osList.includes(resolveRuntimePlatform()) &&
        !remotePlatforms.some((platform) => osList.includes(platform))) {
        return false;
    }
    if (entry.clawdbot?.always === true) {
        return true;
    }
    const requiredBins = entry.clawdbot?.requires?.bins ?? [];
    if (requiredBins.length > 0) {
        for (const bin of requiredBins) {
            if (hasBinary(bin))
                continue;
            if (eligibility?.remote?.hasBin?.(bin))
                continue;
            return false;
        }
    }
    const requiredAnyBins = entry.clawdbot?.requires?.anyBins ?? [];
    if (requiredAnyBins.length > 0) {
        const anyFound = requiredAnyBins.some((bin) => hasBinary(bin)) ||
            eligibility?.remote?.hasAnyBin?.(requiredAnyBins);
        if (!anyFound)
            return false;
    }
    const requiredEnv = entry.clawdbot?.requires?.env ?? [];
    if (requiredEnv.length > 0) {
        for (const envName of requiredEnv) {
            if (process.env[envName])
                continue;
            if (skillConfig?.env?.[envName])
                continue;
            if (skillConfig?.apiKey && entry.clawdbot?.primaryEnv === envName) {
                continue;
            }
            return false;
        }
    }
    const requiredConfig = entry.clawdbot?.requires?.config ?? [];
    if (requiredConfig.length > 0) {
        for (const configPath of requiredConfig) {
            if (!isConfigPathTruthy(config, configPath))
                return false;
        }
    }
    return true;
}
