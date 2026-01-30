export function resolveSlackThreadContext(params) {
    const incomingThreadTs = params.message.thread_ts;
    const eventTs = params.message.event_ts;
    const messageTs = params.message.ts ?? eventTs;
    const hasThreadTs = typeof incomingThreadTs === "string" && incomingThreadTs.length > 0;
    const isThreadReply = hasThreadTs && (incomingThreadTs !== messageTs || Boolean(params.message.parent_user_id));
    const replyToId = incomingThreadTs ?? messageTs;
    const messageThreadId = isThreadReply
        ? incomingThreadTs
        : params.replyToMode === "all"
            ? messageTs
            : undefined;
    return {
        incomingThreadTs,
        messageTs,
        isThreadReply,
        replyToId,
        messageThreadId,
    };
}
export function resolveSlackThreadTargets(params) {
    const { incomingThreadTs, messageTs } = resolveSlackThreadContext(params);
    const replyThreadTs = incomingThreadTs ?? (params.replyToMode === "all" ? messageTs : undefined);
    const statusThreadTs = replyThreadTs ?? messageTs;
    return { replyThreadTs, statusThreadTs };
}
