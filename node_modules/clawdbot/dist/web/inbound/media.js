import { downloadMediaMessage, normalizeMessageContent } from "@whiskeysockets/baileys";
import { logVerbose } from "../../globals.js";
function unwrapMessage(message) {
    const normalized = normalizeMessageContent(message);
    return normalized;
}
export async function downloadInboundMedia(msg, sock) {
    const message = unwrapMessage(msg.message);
    if (!message)
        return undefined;
    const mimetype = message.imageMessage?.mimetype ??
        message.videoMessage?.mimetype ??
        message.documentMessage?.mimetype ??
        message.audioMessage?.mimetype ??
        message.stickerMessage?.mimetype ??
        undefined;
    if (!message.imageMessage &&
        !message.videoMessage &&
        !message.documentMessage &&
        !message.audioMessage &&
        !message.stickerMessage) {
        return undefined;
    }
    try {
        const buffer = (await downloadMediaMessage(msg, "buffer", {}, {
            reuploadRequest: sock.updateMediaMessage,
            logger: sock.logger,
        }));
        return { buffer, mimetype };
    }
    catch (err) {
        logVerbose(`downloadMediaMessage failed: ${String(err)}`);
        return undefined;
    }
}
