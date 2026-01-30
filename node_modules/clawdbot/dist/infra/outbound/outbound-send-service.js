import { dispatchChannelMessageAction } from "../../channels/plugins/message-actions.js";
import { appendAssistantMessageToSessionTranscript } from "../../config/sessions.js";
import { sendMessage, sendPoll } from "./message.js";
function extractToolPayload(result) {
    if (result.details !== undefined)
        return result.details;
    const textBlock = Array.isArray(result.content)
        ? result.content.find((block) => block &&
            typeof block === "object" &&
            block.type === "text" &&
            typeof block.text === "string")
        : undefined;
    const text = textBlock?.text;
    if (text) {
        try {
            return JSON.parse(text);
        }
        catch {
            return text;
        }
    }
    return result.content ?? result;
}
function throwIfAborted(abortSignal) {
    if (abortSignal?.aborted) {
        const err = new Error("Message send aborted");
        err.name = "AbortError";
        throw err;
    }
}
export async function executeSendAction(params) {
    throwIfAborted(params.ctx.abortSignal);
    if (!params.ctx.dryRun) {
        const handled = await dispatchChannelMessageAction({
            channel: params.ctx.channel,
            action: "send",
            cfg: params.ctx.cfg,
            params: params.ctx.params,
            accountId: params.ctx.accountId ?? undefined,
            gateway: params.ctx.gateway,
            toolContext: params.ctx.toolContext,
            dryRun: params.ctx.dryRun,
        });
        if (handled) {
            if (params.ctx.mirror) {
                const mirrorText = params.ctx.mirror.text ?? params.message;
                const mirrorMediaUrls = params.ctx.mirror.mediaUrls ??
                    params.mediaUrls ??
                    (params.mediaUrl ? [params.mediaUrl] : undefined);
                await appendAssistantMessageToSessionTranscript({
                    agentId: params.ctx.mirror.agentId,
                    sessionKey: params.ctx.mirror.sessionKey,
                    text: mirrorText,
                    mediaUrls: mirrorMediaUrls,
                });
            }
            return {
                handledBy: "plugin",
                payload: extractToolPayload(handled),
                toolResult: handled,
            };
        }
    }
    throwIfAborted(params.ctx.abortSignal);
    const result = await sendMessage({
        cfg: params.ctx.cfg,
        to: params.to,
        content: params.message,
        mediaUrl: params.mediaUrl || undefined,
        mediaUrls: params.mediaUrls,
        channel: params.ctx.channel || undefined,
        accountId: params.ctx.accountId ?? undefined,
        gifPlayback: params.gifPlayback,
        dryRun: params.ctx.dryRun,
        bestEffort: params.bestEffort ?? undefined,
        deps: params.ctx.deps,
        gateway: params.ctx.gateway,
        mirror: params.ctx.mirror,
        abortSignal: params.ctx.abortSignal,
    });
    return {
        handledBy: "core",
        payload: result,
        sendResult: result,
    };
}
export async function executePollAction(params) {
    if (!params.ctx.dryRun) {
        const handled = await dispatchChannelMessageAction({
            channel: params.ctx.channel,
            action: "poll",
            cfg: params.ctx.cfg,
            params: params.ctx.params,
            accountId: params.ctx.accountId ?? undefined,
            gateway: params.ctx.gateway,
            toolContext: params.ctx.toolContext,
            dryRun: params.ctx.dryRun,
        });
        if (handled) {
            return {
                handledBy: "plugin",
                payload: extractToolPayload(handled),
                toolResult: handled,
            };
        }
    }
    const result = await sendPoll({
        cfg: params.ctx.cfg,
        to: params.to,
        question: params.question,
        options: params.options,
        maxSelections: params.maxSelections,
        durationHours: params.durationHours ?? undefined,
        channel: params.ctx.channel,
        dryRun: params.ctx.dryRun,
        gateway: params.ctx.gateway,
    });
    return {
        handledBy: "core",
        payload: result,
        pollResult: result,
    };
}
