import { getChannelDock } from "../../channels/dock.js";
import { normalizeChannelId } from "../../channels/plugins/index.js";
export function resolveReplyToMode(cfg, channel, accountId, chatType) {
    const provider = normalizeChannelId(channel);
    if (!provider)
        return "all";
    const resolved = getChannelDock(provider)?.threading?.resolveReplyToMode?.({
        cfg,
        accountId,
        chatType,
    });
    return resolved ?? "all";
}
export function createReplyToModeFilter(mode, opts = {}) {
    let hasThreaded = false;
    return (payload) => {
        if (!payload.replyToId)
            return payload;
        if (mode === "off") {
            if (opts.allowTagsWhenOff && payload.replyToTag)
                return payload;
            return { ...payload, replyToId: undefined };
        }
        if (mode === "all")
            return payload;
        if (hasThreaded) {
            return { ...payload, replyToId: undefined };
        }
        hasThreaded = true;
        return payload;
    };
}
export function createReplyToModeFilterForChannel(mode, channel) {
    const provider = normalizeChannelId(channel);
    const allowTagsWhenOff = provider
        ? Boolean(getChannelDock(provider)?.threading?.allowTagsWhenOff)
        : false;
    return createReplyToModeFilter(mode, {
        allowTagsWhenOff,
    });
}
