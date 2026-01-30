import { buildChannelKeyCandidates, resolveChannelEntryMatchWithFallback, resolveChannelMatchConfig, } from "../../channels/channel-config.js";
import { formatDiscordUserTag } from "./format.js";
export function normalizeDiscordAllowList(raw, prefixes) {
    if (!raw || raw.length === 0)
        return null;
    const ids = new Set();
    const names = new Set();
    const allowAll = raw.some((entry) => String(entry).trim() === "*");
    for (const entry of raw) {
        const text = String(entry).trim();
        if (!text || text === "*")
            continue;
        const normalized = normalizeDiscordSlug(text);
        const maybeId = text.replace(/^<@!?/, "").replace(/>$/, "");
        if (/^\d+$/.test(maybeId)) {
            ids.add(maybeId);
            continue;
        }
        const prefix = prefixes.find((entry) => text.startsWith(entry));
        if (prefix) {
            const candidate = text.slice(prefix.length);
            if (candidate)
                ids.add(candidate);
            continue;
        }
        if (normalized) {
            names.add(normalized);
        }
    }
    return { allowAll, ids, names };
}
export function normalizeDiscordSlug(value) {
    return value
        .trim()
        .toLowerCase()
        .replace(/^#/, "")
        .replace(/[^a-z0-9]+/g, "-")
        .replace(/^-+|-+$/g, "");
}
export function allowListMatches(list, candidate) {
    if (list.allowAll)
        return true;
    if (candidate.id && list.ids.has(candidate.id))
        return true;
    const slug = candidate.name ? normalizeDiscordSlug(candidate.name) : "";
    if (slug && list.names.has(slug))
        return true;
    if (candidate.tag && list.names.has(normalizeDiscordSlug(candidate.tag)))
        return true;
    return false;
}
export function resolveDiscordAllowListMatch(params) {
    const { allowList, candidate } = params;
    if (allowList.allowAll) {
        return { allowed: true, matchKey: "*", matchSource: "wildcard" };
    }
    if (candidate.id && allowList.ids.has(candidate.id)) {
        return { allowed: true, matchKey: candidate.id, matchSource: "id" };
    }
    const nameSlug = candidate.name ? normalizeDiscordSlug(candidate.name) : "";
    if (nameSlug && allowList.names.has(nameSlug)) {
        return { allowed: true, matchKey: nameSlug, matchSource: "name" };
    }
    const tagSlug = candidate.tag ? normalizeDiscordSlug(candidate.tag) : "";
    if (tagSlug && allowList.names.has(tagSlug)) {
        return { allowed: true, matchKey: tagSlug, matchSource: "tag" };
    }
    return { allowed: false };
}
export function resolveDiscordUserAllowed(params) {
    const allowList = normalizeDiscordAllowList(params.allowList, ["discord:", "user:"]);
    if (!allowList)
        return true;
    return allowListMatches(allowList, {
        id: params.userId,
        name: params.userName,
        tag: params.userTag,
    });
}
export function resolveDiscordCommandAuthorized(params) {
    if (!params.isDirectMessage)
        return true;
    const allowList = normalizeDiscordAllowList(params.allowFrom, ["discord:", "user:"]);
    if (!allowList)
        return true;
    return allowListMatches(allowList, {
        id: params.author.id,
        name: params.author.username,
        tag: formatDiscordUserTag(params.author),
    });
}
export function resolveDiscordGuildEntry(params) {
    const guild = params.guild;
    const entries = params.guildEntries;
    if (!guild || !entries)
        return null;
    const byId = entries[guild.id];
    if (byId)
        return { ...byId, id: guild.id };
    const slug = normalizeDiscordSlug(guild.name ?? "");
    const bySlug = entries[slug];
    if (bySlug)
        return { ...bySlug, id: guild.id, slug: slug || bySlug.slug };
    const wildcard = entries["*"];
    if (wildcard)
        return { ...wildcard, id: guild.id, slug: slug || wildcard.slug };
    return null;
}
function buildDiscordChannelKeys(params) {
    const allowNameMatch = params.allowNameMatch !== false;
    return buildChannelKeyCandidates(params.id, allowNameMatch ? params.slug : undefined, allowNameMatch ? params.name : undefined);
}
function resolveDiscordChannelEntryMatch(channels, params, parentParams) {
    const keys = buildDiscordChannelKeys(params);
    const parentKeys = parentParams ? buildDiscordChannelKeys(parentParams) : undefined;
    return resolveChannelEntryMatchWithFallback({
        entries: channels,
        keys,
        parentKeys,
        wildcardKey: "*",
    });
}
function resolveDiscordChannelConfigEntry(entry) {
    const resolved = {
        allowed: entry.allow !== false,
        requireMention: entry.requireMention,
        skills: entry.skills,
        enabled: entry.enabled,
        users: entry.users,
        systemPrompt: entry.systemPrompt,
        autoThread: entry.autoThread,
    };
    return resolved;
}
export function resolveDiscordChannelConfig(params) {
    const { guildInfo, channelId, channelName, channelSlug } = params;
    const channels = guildInfo?.channels;
    if (!channels)
        return null;
    const match = resolveDiscordChannelEntryMatch(channels, {
        id: channelId,
        name: channelName,
        slug: channelSlug,
    });
    const resolved = resolveChannelMatchConfig(match, resolveDiscordChannelConfigEntry);
    return resolved ?? { allowed: false };
}
export function resolveDiscordChannelConfigWithFallback(params) {
    const { guildInfo, channelId, channelName, channelSlug, parentId, parentName, parentSlug, scope, } = params;
    const channels = guildInfo?.channels;
    if (!channels)
        return null;
    const resolvedParentSlug = parentSlug ?? (parentName ? normalizeDiscordSlug(parentName) : "");
    const match = resolveDiscordChannelEntryMatch(channels, {
        id: channelId,
        name: channelName,
        slug: channelSlug,
        allowNameMatch: scope !== "thread",
    }, parentId || parentName || parentSlug
        ? {
            id: parentId ?? "",
            name: parentName,
            slug: resolvedParentSlug,
        }
        : undefined);
    return resolveChannelMatchConfig(match, resolveDiscordChannelConfigEntry) ?? { allowed: false };
}
export function resolveDiscordShouldRequireMention(params) {
    if (!params.isGuildMessage)
        return false;
    // Only skip mention requirement in threads created by the bot (when autoThread is enabled).
    const isBotThread = params.isAutoThreadOwnedByBot ?? isDiscordAutoThreadOwnedByBot(params);
    if (isBotThread)
        return false;
    return params.channelConfig?.requireMention ?? params.guildInfo?.requireMention ?? true;
}
export function isDiscordAutoThreadOwnedByBot(params) {
    if (!params.isThread)
        return false;
    if (!params.channelConfig?.autoThread)
        return false;
    const botId = params.botId?.trim();
    const threadOwnerId = params.threadOwnerId?.trim();
    return Boolean(botId && threadOwnerId && botId === threadOwnerId);
}
export function isDiscordGroupAllowedByPolicy(params) {
    const { groupPolicy, guildAllowlisted, channelAllowlistConfigured, channelAllowed } = params;
    if (groupPolicy === "disabled")
        return false;
    if (groupPolicy === "open")
        return true;
    if (!guildAllowlisted)
        return false;
    if (!channelAllowlistConfigured)
        return true;
    return channelAllowed;
}
export function resolveGroupDmAllow(params) {
    const { channels, channelId, channelName, channelSlug } = params;
    if (!channels || channels.length === 0)
        return true;
    const allowList = channels.map((entry) => normalizeDiscordSlug(String(entry)));
    const candidates = [
        normalizeDiscordSlug(channelId),
        channelSlug,
        channelName ? normalizeDiscordSlug(channelName) : "",
    ].filter(Boolean);
    return allowList.includes("*") || candidates.some((candidate) => allowList.includes(candidate));
}
export function shouldEmitDiscordReactionNotification(params) {
    const mode = params.mode ?? "own";
    if (mode === "off")
        return false;
    if (mode === "all")
        return true;
    if (mode === "own") {
        return Boolean(params.botId && params.messageAuthorId === params.botId);
    }
    if (mode === "allowlist") {
        const list = normalizeDiscordAllowList(params.allowlist, ["discord:", "user:"]);
        if (!list)
            return false;
        return allowListMatches(list, {
            id: params.userId,
            name: params.userName,
            tag: params.userTag,
        });
    }
    return false;
}
