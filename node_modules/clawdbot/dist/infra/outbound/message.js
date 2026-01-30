import { getChannelPlugin, normalizeChannelId } from "../../channels/plugins/index.js";
import { loadConfig } from "../../config/config.js";
import { callGateway, randomIdempotencyKey } from "../../gateway/call.js";
import { normalizePollInput } from "../../polls.js";
import { GATEWAY_CLIENT_MODES, GATEWAY_CLIENT_NAMES, } from "../../utils/message-channel.js";
import { resolveMessageChannelSelection } from "./channel-selection.js";
import { deliverOutboundPayloads, } from "./deliver.js";
import { normalizeReplyPayloadsForDelivery } from "./payloads.js";
import { resolveOutboundTarget } from "./targets.js";
function resolveGatewayOptions(opts) {
    return {
        url: opts?.url,
        token: opts?.token,
        timeoutMs: typeof opts?.timeoutMs === "number" && Number.isFinite(opts.timeoutMs)
            ? Math.max(1, Math.floor(opts.timeoutMs))
            : 10_000,
        clientName: opts?.clientName ?? GATEWAY_CLIENT_NAMES.CLI,
        clientDisplayName: opts?.clientDisplayName,
        mode: opts?.mode ?? GATEWAY_CLIENT_MODES.CLI,
    };
}
export async function sendMessage(params) {
    const cfg = params.cfg ?? loadConfig();
    const channel = params.channel?.trim()
        ? normalizeChannelId(params.channel)
        : (await resolveMessageChannelSelection({ cfg })).channel;
    if (!channel) {
        throw new Error(`Unknown channel: ${params.channel}`);
    }
    const plugin = getChannelPlugin(channel);
    if (!plugin) {
        throw new Error(`Unknown channel: ${channel}`);
    }
    const deliveryMode = plugin.outbound?.deliveryMode ?? "direct";
    const normalizedPayloads = normalizeReplyPayloadsForDelivery([
        {
            text: params.content,
            mediaUrl: params.mediaUrl,
            mediaUrls: params.mediaUrls,
        },
    ]);
    const mirrorText = normalizedPayloads
        .map((payload) => payload.text)
        .filter(Boolean)
        .join("\n");
    const mirrorMediaUrls = normalizedPayloads.flatMap((payload) => payload.mediaUrls ?? (payload.mediaUrl ? [payload.mediaUrl] : []));
    const primaryMediaUrl = mirrorMediaUrls[0] ?? params.mediaUrl ?? null;
    if (params.dryRun) {
        return {
            channel,
            to: params.to,
            via: deliveryMode === "gateway" ? "gateway" : "direct",
            mediaUrl: primaryMediaUrl,
            mediaUrls: mirrorMediaUrls.length ? mirrorMediaUrls : undefined,
            dryRun: true,
        };
    }
    if (deliveryMode !== "gateway") {
        const outboundChannel = channel;
        const resolvedTarget = resolveOutboundTarget({
            channel: outboundChannel,
            to: params.to,
            cfg,
            accountId: params.accountId,
            mode: "explicit",
        });
        if (!resolvedTarget.ok)
            throw resolvedTarget.error;
        const results = await deliverOutboundPayloads({
            cfg,
            channel: outboundChannel,
            to: resolvedTarget.to,
            accountId: params.accountId,
            payloads: normalizedPayloads,
            gifPlayback: params.gifPlayback,
            deps: params.deps,
            bestEffort: params.bestEffort,
            abortSignal: params.abortSignal,
            mirror: params.mirror
                ? {
                    ...params.mirror,
                    text: mirrorText || params.content,
                    mediaUrls: mirrorMediaUrls.length ? mirrorMediaUrls : undefined,
                }
                : undefined,
        });
        return {
            channel,
            to: params.to,
            via: "direct",
            mediaUrl: primaryMediaUrl,
            mediaUrls: mirrorMediaUrls.length ? mirrorMediaUrls : undefined,
            result: results.at(-1),
        };
    }
    const gateway = resolveGatewayOptions(params.gateway);
    const result = await callGateway({
        url: gateway.url,
        token: gateway.token,
        method: "send",
        params: {
            to: params.to,
            message: params.content,
            mediaUrl: params.mediaUrl,
            mediaUrls: mirrorMediaUrls.length ? mirrorMediaUrls : params.mediaUrls,
            gifPlayback: params.gifPlayback,
            accountId: params.accountId,
            channel,
            sessionKey: params.mirror?.sessionKey,
            idempotencyKey: params.idempotencyKey ?? randomIdempotencyKey(),
        },
        timeoutMs: gateway.timeoutMs,
        clientName: gateway.clientName,
        clientDisplayName: gateway.clientDisplayName,
        mode: gateway.mode,
    });
    return {
        channel,
        to: params.to,
        via: "gateway",
        mediaUrl: primaryMediaUrl,
        mediaUrls: mirrorMediaUrls.length ? mirrorMediaUrls : undefined,
        result,
    };
}
export async function sendPoll(params) {
    const cfg = params.cfg ?? loadConfig();
    const channel = params.channel?.trim()
        ? normalizeChannelId(params.channel)
        : (await resolveMessageChannelSelection({ cfg })).channel;
    if (!channel) {
        throw new Error(`Unknown channel: ${params.channel}`);
    }
    const pollInput = {
        question: params.question,
        options: params.options,
        maxSelections: params.maxSelections,
        durationHours: params.durationHours,
    };
    const plugin = getChannelPlugin(channel);
    const outbound = plugin?.outbound;
    if (!outbound?.sendPoll) {
        throw new Error(`Unsupported poll channel: ${channel}`);
    }
    const normalized = outbound.pollMaxOptions
        ? normalizePollInput(pollInput, { maxOptions: outbound.pollMaxOptions })
        : normalizePollInput(pollInput);
    if (params.dryRun) {
        return {
            channel,
            to: params.to,
            question: normalized.question,
            options: normalized.options,
            maxSelections: normalized.maxSelections,
            durationHours: normalized.durationHours ?? null,
            via: "gateway",
            dryRun: true,
        };
    }
    const gateway = resolveGatewayOptions(params.gateway);
    const result = await callGateway({
        url: gateway.url,
        token: gateway.token,
        method: "poll",
        params: {
            to: params.to,
            question: normalized.question,
            options: normalized.options,
            maxSelections: normalized.maxSelections,
            durationHours: normalized.durationHours,
            channel,
            idempotencyKey: params.idempotencyKey ?? randomIdempotencyKey(),
        },
        timeoutMs: gateway.timeoutMs,
        clientName: gateway.clientName,
        clientDisplayName: gateway.clientDisplayName,
        mode: gateway.mode,
    });
    return {
        channel,
        to: params.to,
        question: normalized.question,
        options: normalized.options,
        maxSelections: normalized.maxSelections,
        durationHours: normalized.durationHours ?? null,
        via: "gateway",
        result,
    };
}
