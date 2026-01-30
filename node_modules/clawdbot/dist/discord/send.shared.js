import { RequestClient } from "@buape/carbon";
import { PollLayoutType } from "discord-api-types/payloads/v10";
import { Routes } from "discord-api-types/v10";
import { loadConfig } from "../config/config.js";
import { createDiscordRetryRunner } from "../infra/retry-policy.js";
import { normalizePollDurationHours, normalizePollInput } from "../polls.js";
import { loadWebMedia } from "../web/media.js";
import { resolveDiscordAccount } from "./accounts.js";
import { chunkDiscordTextWithMode } from "./chunk.js";
import { fetchChannelPermissionsDiscord, isThreadChannelType } from "./send.permissions.js";
import { DiscordSendError } from "./send.types.js";
import { parseDiscordTarget } from "./targets.js";
import { normalizeDiscordToken } from "./token.js";
const DISCORD_TEXT_LIMIT = 2000;
const DISCORD_MAX_STICKERS = 3;
const DISCORD_POLL_MAX_ANSWERS = 10;
const DISCORD_POLL_MAX_DURATION_HOURS = 32 * 24;
const DISCORD_MISSING_PERMISSIONS = 50013;
const DISCORD_CANNOT_DM = 50007;
function resolveToken(params) {
    const explicit = normalizeDiscordToken(params.explicit);
    if (explicit)
        return explicit;
    const fallback = normalizeDiscordToken(params.fallbackToken);
    if (!fallback) {
        throw new Error(`Discord bot token missing for account "${params.accountId}" (set discord.accounts.${params.accountId}.token or DISCORD_BOT_TOKEN for default).`);
    }
    return fallback;
}
function resolveRest(token, rest) {
    return rest ?? new RequestClient(token);
}
function createDiscordClient(opts, cfg = loadConfig()) {
    const account = resolveDiscordAccount({ cfg, accountId: opts.accountId });
    const token = resolveToken({
        explicit: opts.token,
        accountId: account.accountId,
        fallbackToken: account.token,
    });
    const rest = resolveRest(token, opts.rest);
    const request = createDiscordRetryRunner({
        retry: opts.retry,
        configRetry: account.config.retry,
        verbose: opts.verbose,
    });
    return { token, rest, request };
}
function resolveDiscordRest(opts) {
    return createDiscordClient(opts).rest;
}
function normalizeReactionEmoji(raw) {
    const trimmed = raw.trim();
    if (!trimmed) {
        throw new Error("emoji required");
    }
    const customMatch = trimmed.match(/^<a?:([^:>]+):(\d+)>$/);
    const identifier = customMatch
        ? `${customMatch[1]}:${customMatch[2]}`
        : trimmed.replace(/[\uFE0E\uFE0F]/g, "");
    return encodeURIComponent(identifier);
}
function parseRecipient(raw) {
    const target = parseDiscordTarget(raw, {
        ambiguousMessage: `Ambiguous Discord recipient "${raw.trim()}". Use "user:${raw.trim()}" for DMs or "channel:${raw.trim()}" for channel messages.`,
    });
    if (!target) {
        throw new Error("Recipient is required for Discord sends");
    }
    return { kind: target.kind, id: target.id };
}
function normalizeStickerIds(raw) {
    const ids = raw.map((entry) => entry.trim()).filter(Boolean);
    if (ids.length === 0) {
        throw new Error("At least one sticker id is required");
    }
    if (ids.length > DISCORD_MAX_STICKERS) {
        throw new Error("Discord supports up to 3 stickers per message");
    }
    return ids;
}
function normalizeEmojiName(raw, label) {
    const name = raw.trim();
    if (!name) {
        throw new Error(`${label} is required`);
    }
    return name;
}
function normalizeDiscordPollInput(input) {
    const poll = normalizePollInput(input, {
        maxOptions: DISCORD_POLL_MAX_ANSWERS,
    });
    const duration = normalizePollDurationHours(poll.durationHours, {
        defaultHours: 24,
        maxHours: DISCORD_POLL_MAX_DURATION_HOURS,
    });
    return {
        question: { text: poll.question },
        answers: poll.options.map((answer) => ({ poll_media: { text: answer } })),
        duration,
        allow_multiselect: poll.maxSelections > 1,
        layout_type: PollLayoutType.Default,
    };
}
function getDiscordErrorCode(err) {
    if (!err || typeof err !== "object")
        return undefined;
    const candidate = "code" in err && err.code !== undefined
        ? err.code
        : "rawError" in err && err.rawError && typeof err.rawError === "object"
            ? err.rawError.code
            : undefined;
    if (typeof candidate === "number")
        return candidate;
    if (typeof candidate === "string" && /^\d+$/.test(candidate)) {
        return Number(candidate);
    }
    return undefined;
}
async function buildDiscordSendError(err, ctx) {
    if (err instanceof DiscordSendError)
        return err;
    const code = getDiscordErrorCode(err);
    if (code === DISCORD_CANNOT_DM) {
        return new DiscordSendError("discord dm failed: user blocks dms or privacy settings disallow it", { kind: "dm-blocked" });
    }
    if (code !== DISCORD_MISSING_PERMISSIONS)
        return err;
    let missing = [];
    try {
        const permissions = await fetchChannelPermissionsDiscord(ctx.channelId, {
            rest: ctx.rest,
            token: ctx.token,
        });
        const current = new Set(permissions.permissions);
        const required = ["ViewChannel", "SendMessages"];
        if (isThreadChannelType(permissions.channelType)) {
            required.push("SendMessagesInThreads");
        }
        if (ctx.hasMedia) {
            required.push("AttachFiles");
        }
        missing = required.filter((permission) => !current.has(permission));
    }
    catch {
        /* ignore permission probe errors */
    }
    const missingLabel = missing.length
        ? `missing permissions in channel ${ctx.channelId}: ${missing.join(", ")}`
        : `missing permissions in channel ${ctx.channelId}`;
    return new DiscordSendError(`${missingLabel}. bot might be muted or blocked by role/channel overrides`, {
        kind: "missing-permissions",
        channelId: ctx.channelId,
        missingPermissions: missing,
    });
}
async function resolveChannelId(rest, recipient, request) {
    if (recipient.kind === "channel") {
        return { channelId: recipient.id };
    }
    const dmChannel = (await request(() => rest.post(Routes.userChannels(), {
        body: { recipient_id: recipient.id },
    }), "dm-channel"));
    if (!dmChannel?.id) {
        throw new Error("Failed to create Discord DM channel");
    }
    return { channelId: dmChannel.id, dm: true };
}
async function sendDiscordText(rest, channelId, text, replyTo, request, maxLinesPerMessage, embeds, chunkMode) {
    if (!text.trim()) {
        throw new Error("Message must be non-empty for Discord sends");
    }
    const messageReference = replyTo ? { message_id: replyTo, fail_if_not_exists: false } : undefined;
    const chunks = chunkDiscordTextWithMode(text, {
        maxChars: DISCORD_TEXT_LIMIT,
        maxLines: maxLinesPerMessage,
        chunkMode,
    });
    if (!chunks.length && text)
        chunks.push(text);
    if (chunks.length === 1) {
        const res = (await request(() => rest.post(Routes.channelMessages(channelId), {
            body: {
                content: chunks[0],
                message_reference: messageReference,
                ...(embeds?.length ? { embeds } : {}),
            },
        }), "text"));
        return res;
    }
    let last = null;
    let isFirst = true;
    for (const chunk of chunks) {
        last = (await request(() => rest.post(Routes.channelMessages(channelId), {
            body: {
                content: chunk,
                message_reference: isFirst ? messageReference : undefined,
                ...(isFirst && embeds?.length ? { embeds } : {}),
            },
        }), "text"));
        isFirst = false;
    }
    if (!last) {
        throw new Error("Discord send failed (empty chunk result)");
    }
    return last;
}
async function sendDiscordMedia(rest, channelId, text, mediaUrl, replyTo, request, maxLinesPerMessage, embeds, chunkMode) {
    const media = await loadWebMedia(mediaUrl);
    const chunks = text
        ? chunkDiscordTextWithMode(text, {
            maxChars: DISCORD_TEXT_LIMIT,
            maxLines: maxLinesPerMessage,
            chunkMode,
        })
        : [];
    if (!chunks.length && text)
        chunks.push(text);
    const caption = chunks[0] ?? "";
    const messageReference = replyTo ? { message_id: replyTo, fail_if_not_exists: false } : undefined;
    const res = (await request(() => rest.post(Routes.channelMessages(channelId), {
        body: {
            content: caption || undefined,
            message_reference: messageReference,
            ...(embeds?.length ? { embeds } : {}),
            files: [
                {
                    data: media.buffer,
                    name: media.fileName ?? "upload",
                },
            ],
        },
    }), "media"));
    for (const chunk of chunks.slice(1)) {
        if (!chunk.trim())
            continue;
        await sendDiscordText(rest, channelId, chunk, undefined, request, maxLinesPerMessage, undefined, chunkMode);
    }
    return res;
}
function buildReactionIdentifier(emoji) {
    if (emoji.id && emoji.name) {
        return `${emoji.name}:${emoji.id}`;
    }
    return emoji.name ?? "";
}
function formatReactionEmoji(emoji) {
    return buildReactionIdentifier(emoji);
}
export { buildDiscordSendError, buildReactionIdentifier, createDiscordClient, formatReactionEmoji, normalizeDiscordPollInput, normalizeEmojiName, normalizeReactionEmoji, normalizeStickerIds, parseRecipient, resolveChannelId, resolveDiscordRest, sendDiscordMedia, sendDiscordText, };
