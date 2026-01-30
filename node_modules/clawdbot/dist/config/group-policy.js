import { normalizeAccountId } from "../routing/session-key.js";
function resolveChannelGroups(cfg, channel, accountId) {
    const normalizedAccountId = normalizeAccountId(accountId);
    const channelConfig = cfg.channels?.[channel];
    if (!channelConfig)
        return undefined;
    const accountGroups = channelConfig.accounts?.[normalizedAccountId]?.groups ??
        channelConfig.accounts?.[Object.keys(channelConfig.accounts ?? {}).find((key) => key.toLowerCase() === normalizedAccountId.toLowerCase()) ?? ""]?.groups;
    return accountGroups ?? channelConfig.groups;
}
export function resolveChannelGroupPolicy(params) {
    const { cfg, channel } = params;
    const groups = resolveChannelGroups(cfg, channel, params.accountId);
    const allowlistEnabled = Boolean(groups && Object.keys(groups).length > 0);
    const normalizedId = params.groupId?.trim();
    const groupConfig = normalizedId && groups ? groups[normalizedId] : undefined;
    const defaultConfig = groups?.["*"];
    const allowAll = allowlistEnabled && Boolean(groups && Object.hasOwn(groups, "*"));
    const allowed = !allowlistEnabled ||
        allowAll ||
        (normalizedId ? Boolean(groups && Object.hasOwn(groups, normalizedId)) : false);
    return {
        allowlistEnabled,
        allowed,
        groupConfig,
        defaultConfig,
    };
}
export function resolveChannelGroupRequireMention(params) {
    const { requireMentionOverride, overrideOrder = "after-config" } = params;
    const { groupConfig, defaultConfig } = resolveChannelGroupPolicy(params);
    const configMention = typeof groupConfig?.requireMention === "boolean"
        ? groupConfig.requireMention
        : typeof defaultConfig?.requireMention === "boolean"
            ? defaultConfig.requireMention
            : undefined;
    if (overrideOrder === "before-config" && typeof requireMentionOverride === "boolean") {
        return requireMentionOverride;
    }
    if (typeof configMention === "boolean")
        return configMention;
    if (overrideOrder !== "before-config" && typeof requireMentionOverride === "boolean") {
        return requireMentionOverride;
    }
    return true;
}
export function resolveChannelGroupToolsPolicy(params) {
    const { groupConfig, defaultConfig } = resolveChannelGroupPolicy(params);
    if (groupConfig?.tools)
        return groupConfig.tools;
    if (defaultConfig?.tools)
        return defaultConfig.tools;
    return undefined;
}
