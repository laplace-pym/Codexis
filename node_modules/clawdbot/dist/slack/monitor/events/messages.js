import { danger } from "../../../globals.js";
import { enqueueSystemEvent } from "../../../infra/system-events.js";
import { resolveSlackChannelLabel } from "../channel-config.js";
export function registerSlackMessageEvents(params) {
    const { ctx, handleSlackMessage } = params;
    ctx.app.event("message", async ({ event, body }) => {
        try {
            if (ctx.shouldDropMismatchedSlackEvent(body))
                return;
            const message = event;
            if (message.subtype === "message_changed") {
                const changed = event;
                const channelId = changed.channel;
                const channelInfo = channelId ? await ctx.resolveChannelName(channelId) : {};
                const channelType = channelInfo?.type;
                if (!ctx.isChannelAllowed({
                    channelId,
                    channelName: channelInfo?.name,
                    channelType,
                })) {
                    return;
                }
                const messageId = changed.message?.ts ?? changed.previous_message?.ts;
                const label = resolveSlackChannelLabel({
                    channelId,
                    channelName: channelInfo?.name,
                });
                const sessionKey = ctx.resolveSlackSystemEventSessionKey({
                    channelId,
                    channelType,
                });
                enqueueSystemEvent(`Slack message edited in ${label}.`, {
                    sessionKey,
                    contextKey: `slack:message:changed:${channelId ?? "unknown"}:${messageId ?? changed.event_ts ?? "unknown"}`,
                });
                return;
            }
            if (message.subtype === "message_deleted") {
                const deleted = event;
                const channelId = deleted.channel;
                const channelInfo = channelId ? await ctx.resolveChannelName(channelId) : {};
                const channelType = channelInfo?.type;
                if (!ctx.isChannelAllowed({
                    channelId,
                    channelName: channelInfo?.name,
                    channelType,
                })) {
                    return;
                }
                const label = resolveSlackChannelLabel({
                    channelId,
                    channelName: channelInfo?.name,
                });
                const sessionKey = ctx.resolveSlackSystemEventSessionKey({
                    channelId,
                    channelType,
                });
                enqueueSystemEvent(`Slack message deleted in ${label}.`, {
                    sessionKey,
                    contextKey: `slack:message:deleted:${channelId ?? "unknown"}:${deleted.deleted_ts ?? deleted.event_ts ?? "unknown"}`,
                });
                return;
            }
            if (message.subtype === "thread_broadcast") {
                const thread = event;
                const channelId = thread.channel;
                const channelInfo = channelId ? await ctx.resolveChannelName(channelId) : {};
                const channelType = channelInfo?.type;
                if (!ctx.isChannelAllowed({
                    channelId,
                    channelName: channelInfo?.name,
                    channelType,
                })) {
                    return;
                }
                const label = resolveSlackChannelLabel({
                    channelId,
                    channelName: channelInfo?.name,
                });
                const messageId = thread.message?.ts ?? thread.event_ts;
                const sessionKey = ctx.resolveSlackSystemEventSessionKey({
                    channelId,
                    channelType,
                });
                enqueueSystemEvent(`Slack thread reply broadcast in ${label}.`, {
                    sessionKey,
                    contextKey: `slack:thread:broadcast:${channelId ?? "unknown"}:${messageId ?? "unknown"}`,
                });
                return;
            }
            await handleSlackMessage(message, { source: "message" });
        }
        catch (err) {
            ctx.runtime.error?.(danger(`slack handler failed: ${String(err)}`));
        }
    });
    ctx.app.event("app_mention", async ({ event, body }) => {
        try {
            if (ctx.shouldDropMismatchedSlackEvent(body))
                return;
            const mention = event;
            await handleSlackMessage(mention, {
                source: "app_mention",
                wasMentioned: true,
            });
        }
        catch (err) {
            ctx.runtime.error?.(danger(`slack mention handler failed: ${String(err)}`));
        }
    });
}
