import { formatAgentEnvelope } from "../../auto-reply/envelope.js";
import { formatDiscordUserTag, resolveTimestampMs } from "./format.js";
export function resolveReplyContext(message, resolveDiscordMessageText, options) {
    const referenced = message.referencedMessage;
    if (!referenced?.author)
        return null;
    const referencedText = resolveDiscordMessageText(referenced, {
        includeForwarded: true,
    });
    if (!referencedText)
        return null;
    const fromLabel = referenced.author ? buildDirectLabel(referenced.author) : "Unknown";
    const body = `${referencedText}\n[discord message id: ${referenced.id} channel: ${referenced.channelId} from: ${formatDiscordUserTag(referenced.author)} user id:${referenced.author?.id ?? "unknown"}]`;
    return formatAgentEnvelope({
        channel: "Discord",
        from: fromLabel,
        timestamp: resolveTimestampMs(referenced.timestamp),
        body,
        envelope: options?.envelope,
    });
}
export function buildDirectLabel(author) {
    const username = formatDiscordUserTag(author);
    return `${username} user id:${author.id}`;
}
export function buildGuildLabel(params) {
    const { guild, channelName, channelId } = params;
    return `${guild?.name ?? "Guild"} #${channelName} channel id:${channelId}`;
}
