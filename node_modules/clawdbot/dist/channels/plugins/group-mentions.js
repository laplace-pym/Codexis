import { resolveChannelGroupRequireMention, resolveChannelGroupToolsPolicy, } from "../../config/group-policy.js";
import { resolveSlackAccount } from "../../slack/accounts.js";
function normalizeDiscordSlug(value) {
    if (!value)
        return "";
    let text = value.trim().toLowerCase();
    if (!text)
        return "";
    text = text.replace(/^[@#]+/, "");
    text = text.replace(/[\s_]+/g, "-");
    text = text.replace(/[^a-z0-9-]+/g, "-");
    text = text.replace(/-{2,}/g, "-").replace(/^-+|-+$/g, "");
    return text;
}
function normalizeSlackSlug(raw) {
    const trimmed = raw?.trim().toLowerCase() ?? "";
    if (!trimmed)
        return "";
    const dashed = trimmed.replace(/\s+/g, "-");
    const cleaned = dashed.replace(/[^a-z0-9#@._+-]+/g, "-");
    return cleaned.replace(/-{2,}/g, "-").replace(/^[-.]+|[-.]+$/g, "");
}
function parseTelegramGroupId(value) {
    const raw = value?.trim() ?? "";
    if (!raw)
        return { chatId: undefined, topicId: undefined };
    const parts = raw.split(":").filter(Boolean);
    if (parts.length >= 3 &&
        parts[1] === "topic" &&
        /^-?\d+$/.test(parts[0]) &&
        /^\d+$/.test(parts[2])) {
        return { chatId: parts[0], topicId: parts[2] };
    }
    if (parts.length >= 2 && /^-?\d+$/.test(parts[0]) && /^\d+$/.test(parts[1])) {
        return { chatId: parts[0], topicId: parts[1] };
    }
    return { chatId: raw, topicId: undefined };
}
function resolveTelegramRequireMention(params) {
    const { cfg, chatId, topicId } = params;
    if (!chatId)
        return undefined;
    const groupConfig = cfg.channels?.telegram?.groups?.[chatId];
    const groupDefault = cfg.channels?.telegram?.groups?.["*"];
    const topicConfig = topicId && groupConfig?.topics ? groupConfig.topics[topicId] : undefined;
    const defaultTopicConfig = topicId && groupDefault?.topics ? groupDefault.topics[topicId] : undefined;
    if (typeof topicConfig?.requireMention === "boolean") {
        return topicConfig.requireMention;
    }
    if (typeof defaultTopicConfig?.requireMention === "boolean") {
        return defaultTopicConfig.requireMention;
    }
    if (typeof groupConfig?.requireMention === "boolean") {
        return groupConfig.requireMention;
    }
    if (typeof groupDefault?.requireMention === "boolean") {
        return groupDefault.requireMention;
    }
    return undefined;
}
function resolveDiscordGuildEntry(guilds, groupSpace) {
    if (!guilds || Object.keys(guilds).length === 0)
        return null;
    const space = groupSpace?.trim() ?? "";
    if (space && guilds[space])
        return guilds[space];
    const normalized = normalizeDiscordSlug(space);
    if (normalized && guilds[normalized])
        return guilds[normalized];
    if (normalized) {
        const match = Object.values(guilds).find((entry) => normalizeDiscordSlug(entry?.slug ?? undefined) === normalized);
        if (match)
            return match;
    }
    return guilds["*"] ?? null;
}
export function resolveTelegramGroupRequireMention(params) {
    const { chatId, topicId } = parseTelegramGroupId(params.groupId);
    const requireMention = resolveTelegramRequireMention({
        cfg: params.cfg,
        chatId,
        topicId,
    });
    if (typeof requireMention === "boolean")
        return requireMention;
    return resolveChannelGroupRequireMention({
        cfg: params.cfg,
        channel: "telegram",
        groupId: chatId ?? params.groupId,
        accountId: params.accountId,
    });
}
export function resolveWhatsAppGroupRequireMention(params) {
    return resolveChannelGroupRequireMention({
        cfg: params.cfg,
        channel: "whatsapp",
        groupId: params.groupId,
        accountId: params.accountId,
    });
}
export function resolveIMessageGroupRequireMention(params) {
    return resolveChannelGroupRequireMention({
        cfg: params.cfg,
        channel: "imessage",
        groupId: params.groupId,
        accountId: params.accountId,
    });
}
export function resolveDiscordGroupRequireMention(params) {
    const guildEntry = resolveDiscordGuildEntry(params.cfg.channels?.discord?.guilds, params.groupSpace);
    const channelEntries = guildEntry?.channels;
    if (channelEntries && Object.keys(channelEntries).length > 0) {
        const groupChannel = params.groupChannel;
        const channelSlug = normalizeDiscordSlug(groupChannel);
        const entry = (params.groupId ? channelEntries[params.groupId] : undefined) ??
            (channelSlug
                ? (channelEntries[channelSlug] ?? channelEntries[`#${channelSlug}`])
                : undefined) ??
            (groupChannel ? channelEntries[normalizeDiscordSlug(groupChannel)] : undefined);
        if (entry && typeof entry.requireMention === "boolean") {
            return entry.requireMention;
        }
    }
    if (typeof guildEntry?.requireMention === "boolean") {
        return guildEntry.requireMention;
    }
    return true;
}
export function resolveGoogleChatGroupRequireMention(params) {
    return resolveChannelGroupRequireMention({
        cfg: params.cfg,
        channel: "googlechat",
        groupId: params.groupId,
        accountId: params.accountId,
    });
}
export function resolveGoogleChatGroupToolPolicy(params) {
    return resolveChannelGroupToolsPolicy({
        cfg: params.cfg,
        channel: "googlechat",
        groupId: params.groupId,
        accountId: params.accountId,
    });
}
export function resolveSlackGroupRequireMention(params) {
    const account = resolveSlackAccount({
        cfg: params.cfg,
        accountId: params.accountId,
    });
    const channels = account.channels ?? {};
    const keys = Object.keys(channels);
    if (keys.length === 0)
        return true;
    const channelId = params.groupId?.trim();
    const groupChannel = params.groupChannel;
    const channelName = groupChannel?.replace(/^#/, "");
    const normalizedName = normalizeSlackSlug(channelName);
    const candidates = [
        channelId ?? "",
        channelName ? `#${channelName}` : "",
        channelName ?? "",
        normalizedName,
    ].filter(Boolean);
    let matched;
    for (const candidate of candidates) {
        if (candidate && channels[candidate]) {
            matched = channels[candidate];
            break;
        }
    }
    const fallback = channels["*"];
    const resolved = matched ?? fallback;
    if (typeof resolved?.requireMention === "boolean") {
        return resolved.requireMention;
    }
    return true;
}
export function resolveBlueBubblesGroupRequireMention(params) {
    return resolveChannelGroupRequireMention({
        cfg: params.cfg,
        channel: "bluebubbles",
        groupId: params.groupId,
        accountId: params.accountId,
    });
}
export function resolveTelegramGroupToolPolicy(params) {
    const { chatId } = parseTelegramGroupId(params.groupId);
    return resolveChannelGroupToolsPolicy({
        cfg: params.cfg,
        channel: "telegram",
        groupId: chatId ?? params.groupId,
        accountId: params.accountId,
    });
}
export function resolveWhatsAppGroupToolPolicy(params) {
    return resolveChannelGroupToolsPolicy({
        cfg: params.cfg,
        channel: "whatsapp",
        groupId: params.groupId,
        accountId: params.accountId,
    });
}
export function resolveIMessageGroupToolPolicy(params) {
    return resolveChannelGroupToolsPolicy({
        cfg: params.cfg,
        channel: "imessage",
        groupId: params.groupId,
        accountId: params.accountId,
    });
}
export function resolveDiscordGroupToolPolicy(params) {
    const guildEntry = resolveDiscordGuildEntry(params.cfg.channels?.discord?.guilds, params.groupSpace);
    const channelEntries = guildEntry?.channels;
    if (channelEntries && Object.keys(channelEntries).length > 0) {
        const groupChannel = params.groupChannel;
        const channelSlug = normalizeDiscordSlug(groupChannel);
        const entry = (params.groupId ? channelEntries[params.groupId] : undefined) ??
            (channelSlug
                ? (channelEntries[channelSlug] ?? channelEntries[`#${channelSlug}`])
                : undefined) ??
            (groupChannel ? channelEntries[normalizeDiscordSlug(groupChannel)] : undefined);
        if (entry?.tools)
            return entry.tools;
    }
    if (guildEntry?.tools)
        return guildEntry.tools;
    return undefined;
}
export function resolveSlackGroupToolPolicy(params) {
    const account = resolveSlackAccount({
        cfg: params.cfg,
        accountId: params.accountId,
    });
    const channels = account.channels ?? {};
    const keys = Object.keys(channels);
    if (keys.length === 0)
        return undefined;
    const channelId = params.groupId?.trim();
    const groupChannel = params.groupChannel;
    const channelName = groupChannel?.replace(/^#/, "");
    const normalizedName = normalizeSlackSlug(channelName);
    const candidates = [
        channelId ?? "",
        channelName ? `#${channelName}` : "",
        channelName ?? "",
        normalizedName,
    ].filter(Boolean);
    let matched;
    for (const candidate of candidates) {
        if (candidate && channels[candidate]) {
            matched = channels[candidate];
            break;
        }
    }
    const resolved = matched ?? channels["*"];
    if (resolved?.tools)
        return resolved.tools;
    return undefined;
}
export function resolveBlueBubblesGroupToolPolicy(params) {
    return resolveChannelGroupToolsPolicy({
        cfg: params.cfg,
        channel: "bluebubbles",
        groupId: params.groupId,
        accountId: params.accountId,
    });
}
