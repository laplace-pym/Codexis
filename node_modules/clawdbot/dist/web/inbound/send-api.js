import { recordChannelActivity } from "../../infra/channel-activity.js";
import { toWhatsappJid } from "../../utils.js";
export function createWebSendApi(params) {
    return {
        sendMessage: async (to, text, mediaBuffer, mediaType, sendOptions) => {
            const jid = toWhatsappJid(to);
            let payload;
            if (mediaBuffer && mediaType) {
                if (mediaType.startsWith("image/")) {
                    payload = {
                        image: mediaBuffer,
                        caption: text || undefined,
                        mimetype: mediaType,
                    };
                }
                else if (mediaType.startsWith("audio/")) {
                    payload = { audio: mediaBuffer, ptt: true, mimetype: mediaType };
                }
                else if (mediaType.startsWith("video/")) {
                    const gifPlayback = sendOptions?.gifPlayback;
                    payload = {
                        video: mediaBuffer,
                        caption: text || undefined,
                        mimetype: mediaType,
                        ...(gifPlayback ? { gifPlayback: true } : {}),
                    };
                }
                else {
                    payload = {
                        document: mediaBuffer,
                        fileName: "file",
                        caption: text || undefined,
                        mimetype: mediaType,
                    };
                }
            }
            else {
                payload = { text };
            }
            const result = await params.sock.sendMessage(jid, payload);
            const accountId = sendOptions?.accountId ?? params.defaultAccountId;
            recordChannelActivity({
                channel: "whatsapp",
                accountId,
                direction: "outbound",
            });
            const messageId = typeof result === "object" && result && "key" in result
                ? String(result.key?.id ?? "unknown")
                : "unknown";
            return { messageId };
        },
        sendPoll: async (to, poll) => {
            const jid = toWhatsappJid(to);
            const result = await params.sock.sendMessage(jid, {
                poll: {
                    name: poll.question,
                    values: poll.options,
                    selectableCount: poll.maxSelections ?? 1,
                },
            });
            recordChannelActivity({
                channel: "whatsapp",
                accountId: params.defaultAccountId,
                direction: "outbound",
            });
            const messageId = typeof result === "object" && result && "key" in result
                ? String(result.key?.id ?? "unknown")
                : "unknown";
            return { messageId };
        },
        sendReaction: async (chatJid, messageId, emoji, fromMe, participant) => {
            const jid = toWhatsappJid(chatJid);
            await params.sock.sendMessage(jid, {
                react: {
                    text: emoji,
                    key: {
                        remoteJid: jid,
                        id: messageId,
                        fromMe,
                        participant: participant ? toWhatsappJid(participant) : undefined,
                    },
                },
            });
        },
        sendComposingTo: async (to) => {
            const jid = toWhatsappJid(to);
            await params.sock.sendPresenceUpdate("composing", jid);
        },
    };
}
