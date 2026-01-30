import { chunkTextWithMode, resolveChunkMode } from "../../auto-reply/chunk.js";
import { loadConfig } from "../../config/config.js";
import { resolveMarkdownTableMode } from "../../config/markdown-tables.js";
import { convertMarkdownTables } from "../../markdown/tables.js";
import { sendMessageIMessage } from "../send.js";
export async function deliverReplies(params) {
    const { replies, target, client, runtime, maxBytes, textLimit, accountId } = params;
    const cfg = loadConfig();
    const tableMode = resolveMarkdownTableMode({
        cfg,
        channel: "imessage",
        accountId,
    });
    const chunkMode = resolveChunkMode(cfg, "imessage", accountId);
    for (const payload of replies) {
        const mediaList = payload.mediaUrls ?? (payload.mediaUrl ? [payload.mediaUrl] : []);
        const rawText = payload.text ?? "";
        const text = convertMarkdownTables(rawText, tableMode);
        if (!text && mediaList.length === 0)
            continue;
        if (mediaList.length === 0) {
            for (const chunk of chunkTextWithMode(text, textLimit, chunkMode)) {
                await sendMessageIMessage(target, chunk, {
                    maxBytes,
                    client,
                    accountId,
                });
            }
        }
        else {
            let first = true;
            for (const url of mediaList) {
                const caption = first ? text : "";
                first = false;
                await sendMessageIMessage(target, caption, {
                    mediaUrl: url,
                    maxBytes,
                    client,
                    accountId,
                });
            }
        }
        runtime.log?.(`imessage: delivered reply to ${target}`);
    }
}
