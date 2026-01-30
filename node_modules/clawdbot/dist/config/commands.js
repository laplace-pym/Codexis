import { normalizeChannelId } from "../channels/plugins/index.js";
function resolveAutoDefault(providerId) {
    const id = normalizeChannelId(providerId);
    if (!id)
        return false;
    if (id === "discord" || id === "telegram")
        return true;
    if (id === "slack")
        return false;
    return false;
}
export function resolveNativeSkillsEnabled(params) {
    const { providerId, providerSetting, globalSetting } = params;
    const setting = providerSetting === undefined ? globalSetting : providerSetting;
    if (setting === true)
        return true;
    if (setting === false)
        return false;
    return resolveAutoDefault(providerId);
}
export function resolveNativeCommandsEnabled(params) {
    const { providerId, providerSetting, globalSetting } = params;
    const setting = providerSetting === undefined ? globalSetting : providerSetting;
    if (setting === true)
        return true;
    if (setting === false)
        return false;
    // auto or undefined -> heuristic
    return resolveAutoDefault(providerId);
}
export function isNativeCommandsExplicitlyDisabled(params) {
    const { providerSetting, globalSetting } = params;
    if (providerSetting === false)
        return true;
    if (providerSetting === undefined)
        return globalSetting === false;
    return false;
}
