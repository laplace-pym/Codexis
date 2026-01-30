import { recordSessionMetaFromInbound, updateLastRoute, } from "../config/sessions.js";
export async function recordInboundSession(params) {
    const { storePath, sessionKey, ctx, groupResolution, createIfMissing } = params;
    void recordSessionMetaFromInbound({
        storePath,
        sessionKey,
        ctx,
        groupResolution,
        createIfMissing,
    }).catch(params.onRecordError);
    const update = params.updateLastRoute;
    if (!update)
        return;
    await updateLastRoute({
        storePath,
        sessionKey: update.sessionKey,
        deliveryContext: {
            channel: update.channel,
            to: update.to,
            accountId: update.accountId,
            threadId: update.threadId,
        },
        ctx,
        groupResolution,
    });
}
