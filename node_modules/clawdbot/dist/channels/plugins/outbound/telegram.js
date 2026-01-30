import { markdownToTelegramHtmlChunks } from "../../../telegram/format.js";
import { sendMessageTelegram } from "../../../telegram/send.js";
function parseReplyToMessageId(replyToId) {
    if (!replyToId)
        return undefined;
    const parsed = Number.parseInt(replyToId, 10);
    return Number.isFinite(parsed) ? parsed : undefined;
}
function parseThreadId(threadId) {
    if (threadId == null)
        return undefined;
    if (typeof threadId === "number") {
        return Number.isFinite(threadId) ? Math.trunc(threadId) : undefined;
    }
    const trimmed = threadId.trim();
    if (!trimmed)
        return undefined;
    const parsed = Number.parseInt(trimmed, 10);
    return Number.isFinite(parsed) ? parsed : undefined;
}
export const telegramOutbound = {
    deliveryMode: "direct",
    chunker: markdownToTelegramHtmlChunks,
    chunkerMode: "markdown",
    textChunkLimit: 4000,
    sendText: async ({ to, text, accountId, deps, replyToId, threadId }) => {
        const send = deps?.sendTelegram ?? sendMessageTelegram;
        const replyToMessageId = parseReplyToMessageId(replyToId);
        const messageThreadId = parseThreadId(threadId);
        const result = await send(to, text, {
            verbose: false,
            textMode: "html",
            messageThreadId,
            replyToMessageId,
            accountId: accountId ?? undefined,
        });
        return { channel: "telegram", ...result };
    },
    sendMedia: async ({ to, text, mediaUrl, accountId, deps, replyToId, threadId }) => {
        const send = deps?.sendTelegram ?? sendMessageTelegram;
        const replyToMessageId = parseReplyToMessageId(replyToId);
        const messageThreadId = parseThreadId(threadId);
        const result = await send(to, text, {
            verbose: false,
            mediaUrl,
            textMode: "html",
            messageThreadId,
            replyToMessageId,
            accountId: accountId ?? undefined,
        });
        return { channel: "telegram", ...result };
    },
};
