import { chunkByParagraph, chunkMarkdownTextWithMode, resolveChunkMode, resolveTextChunkLimit, } from "../../auto-reply/chunk.js";
import { resolveChannelMediaMaxBytes } from "../../channels/plugins/media-limits.js";
import { loadChannelOutboundAdapter } from "../../channels/plugins/outbound/load.js";
import { resolveMarkdownTableMode } from "../../config/markdown-tables.js";
import { markdownToSignalTextChunks } from "../../signal/format.js";
import { sendMessageSignal } from "../../signal/send.js";
import { appendAssistantMessageToSessionTranscript, resolveMirroredTranscriptText, } from "../../config/sessions.js";
import { normalizeReplyPayloadsForDelivery } from "./payloads.js";
export { normalizeOutboundPayloads } from "./payloads.js";
function throwIfAborted(abortSignal) {
    if (abortSignal?.aborted) {
        throw new Error("Outbound delivery aborted");
    }
}
// Channel docking: outbound delivery delegates to plugin.outbound adapters.
async function createChannelHandler(params) {
    const outbound = await loadChannelOutboundAdapter(params.channel);
    if (!outbound?.sendText || !outbound?.sendMedia) {
        throw new Error(`Outbound not configured for channel: ${params.channel}`);
    }
    const handler = createPluginHandler({
        outbound,
        cfg: params.cfg,
        channel: params.channel,
        to: params.to,
        accountId: params.accountId,
        replyToId: params.replyToId,
        threadId: params.threadId,
        deps: params.deps,
        gifPlayback: params.gifPlayback,
    });
    if (!handler) {
        throw new Error(`Outbound not configured for channel: ${params.channel}`);
    }
    return handler;
}
function createPluginHandler(params) {
    const outbound = params.outbound;
    if (!outbound?.sendText || !outbound?.sendMedia)
        return null;
    const sendText = outbound.sendText;
    const sendMedia = outbound.sendMedia;
    const chunker = outbound.chunker ?? null;
    const chunkerMode = outbound.chunkerMode;
    return {
        chunker,
        chunkerMode,
        textChunkLimit: outbound.textChunkLimit,
        sendPayload: outbound.sendPayload
            ? async (payload) => outbound.sendPayload({
                cfg: params.cfg,
                to: params.to,
                text: payload.text ?? "",
                mediaUrl: payload.mediaUrl,
                accountId: params.accountId,
                replyToId: params.replyToId,
                threadId: params.threadId,
                gifPlayback: params.gifPlayback,
                deps: params.deps,
                payload,
            })
            : undefined,
        sendText: async (text) => sendText({
            cfg: params.cfg,
            to: params.to,
            text,
            accountId: params.accountId,
            replyToId: params.replyToId,
            threadId: params.threadId,
            gifPlayback: params.gifPlayback,
            deps: params.deps,
        }),
        sendMedia: async (caption, mediaUrl) => sendMedia({
            cfg: params.cfg,
            to: params.to,
            text: caption,
            mediaUrl,
            accountId: params.accountId,
            replyToId: params.replyToId,
            threadId: params.threadId,
            gifPlayback: params.gifPlayback,
            deps: params.deps,
        }),
    };
}
export async function deliverOutboundPayloads(params) {
    const { cfg, channel, to, payloads } = params;
    const accountId = params.accountId;
    const deps = params.deps;
    const abortSignal = params.abortSignal;
    const sendSignal = params.deps?.sendSignal ?? sendMessageSignal;
    const results = [];
    const handler = await createChannelHandler({
        cfg,
        channel,
        to,
        deps,
        accountId,
        replyToId: params.replyToId,
        threadId: params.threadId,
        gifPlayback: params.gifPlayback,
    });
    const textLimit = handler.chunker
        ? resolveTextChunkLimit(cfg, channel, accountId, {
            fallbackLimit: handler.textChunkLimit,
        })
        : undefined;
    const chunkMode = handler.chunker ? resolveChunkMode(cfg, channel, accountId) : "length";
    const isSignalChannel = channel === "signal";
    const signalTableMode = isSignalChannel
        ? resolveMarkdownTableMode({ cfg, channel: "signal", accountId })
        : "code";
    const signalMaxBytes = isSignalChannel
        ? resolveChannelMediaMaxBytes({
            cfg,
            resolveChannelLimitMb: ({ cfg, accountId }) => cfg.channels?.signal?.accounts?.[accountId]?.mediaMaxMb ??
                cfg.channels?.signal?.mediaMaxMb,
            accountId,
        })
        : undefined;
    const sendTextChunks = async (text) => {
        throwIfAborted(abortSignal);
        if (!handler.chunker || textLimit === undefined) {
            results.push(await handler.sendText(text));
            return;
        }
        if (chunkMode === "newline") {
            const mode = handler.chunkerMode ?? "text";
            const blockChunks = mode === "markdown"
                ? chunkMarkdownTextWithMode(text, textLimit, "newline")
                : chunkByParagraph(text, textLimit);
            if (!blockChunks.length && text)
                blockChunks.push(text);
            for (const blockChunk of blockChunks) {
                const chunks = handler.chunker(blockChunk, textLimit);
                if (!chunks.length && blockChunk)
                    chunks.push(blockChunk);
                for (const chunk of chunks) {
                    throwIfAborted(abortSignal);
                    results.push(await handler.sendText(chunk));
                }
            }
            return;
        }
        const chunks = handler.chunker(text, textLimit);
        for (const chunk of chunks) {
            throwIfAborted(abortSignal);
            results.push(await handler.sendText(chunk));
        }
    };
    const sendSignalText = async (text, styles) => {
        throwIfAborted(abortSignal);
        return {
            channel: "signal",
            ...(await sendSignal(to, text, {
                maxBytes: signalMaxBytes,
                accountId: accountId ?? undefined,
                textMode: "plain",
                textStyles: styles,
            })),
        };
    };
    const sendSignalTextChunks = async (text) => {
        throwIfAborted(abortSignal);
        let signalChunks = textLimit === undefined
            ? markdownToSignalTextChunks(text, Number.POSITIVE_INFINITY, {
                tableMode: signalTableMode,
            })
            : markdownToSignalTextChunks(text, textLimit, { tableMode: signalTableMode });
        if (signalChunks.length === 0 && text) {
            signalChunks = [{ text, styles: [] }];
        }
        for (const chunk of signalChunks) {
            throwIfAborted(abortSignal);
            results.push(await sendSignalText(chunk.text, chunk.styles));
        }
    };
    const sendSignalMedia = async (caption, mediaUrl) => {
        throwIfAborted(abortSignal);
        const formatted = markdownToSignalTextChunks(caption, Number.POSITIVE_INFINITY, {
            tableMode: signalTableMode,
        })[0] ?? {
            text: caption,
            styles: [],
        };
        return {
            channel: "signal",
            ...(await sendSignal(to, formatted.text, {
                mediaUrl,
                maxBytes: signalMaxBytes,
                accountId: accountId ?? undefined,
                textMode: "plain",
                textStyles: formatted.styles,
            })),
        };
    };
    const normalizedPayloads = normalizeReplyPayloadsForDelivery(payloads);
    for (const payload of normalizedPayloads) {
        const payloadSummary = {
            text: payload.text ?? "",
            mediaUrls: payload.mediaUrls ?? (payload.mediaUrl ? [payload.mediaUrl] : []),
            channelData: payload.channelData,
        };
        try {
            throwIfAborted(abortSignal);
            params.onPayload?.(payloadSummary);
            if (handler.sendPayload && payload.channelData) {
                results.push(await handler.sendPayload(payload));
                continue;
            }
            if (payloadSummary.mediaUrls.length === 0) {
                if (isSignalChannel) {
                    await sendSignalTextChunks(payloadSummary.text);
                }
                else {
                    await sendTextChunks(payloadSummary.text);
                }
                continue;
            }
            let first = true;
            for (const url of payloadSummary.mediaUrls) {
                throwIfAborted(abortSignal);
                const caption = first ? payloadSummary.text : "";
                first = false;
                if (isSignalChannel) {
                    results.push(await sendSignalMedia(caption, url));
                }
                else {
                    results.push(await handler.sendMedia(caption, url));
                }
            }
        }
        catch (err) {
            if (!params.bestEffort)
                throw err;
            params.onError?.(err, payloadSummary);
        }
    }
    if (params.mirror && results.length > 0) {
        const mirrorText = resolveMirroredTranscriptText({
            text: params.mirror.text,
            mediaUrls: params.mirror.mediaUrls,
        });
        if (mirrorText) {
            await appendAssistantMessageToSessionTranscript({
                agentId: params.mirror.agentId,
                sessionKey: params.mirror.sessionKey,
                text: mirrorText,
            });
        }
    }
    return results;
}
