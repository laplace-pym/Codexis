import { finalizeInboundContext } from "./reply/inbound-context.js";
import { dispatchReplyFromConfig } from "./reply/dispatch-from-config.js";
import { createReplyDispatcher, createReplyDispatcherWithTyping, } from "./reply/reply-dispatcher.js";
export async function dispatchInboundMessage(params) {
    const finalized = finalizeInboundContext(params.ctx);
    return await dispatchReplyFromConfig({
        ctx: finalized,
        cfg: params.cfg,
        dispatcher: params.dispatcher,
        replyOptions: params.replyOptions,
        replyResolver: params.replyResolver,
    });
}
export async function dispatchInboundMessageWithBufferedDispatcher(params) {
    const { dispatcher, replyOptions, markDispatchIdle } = createReplyDispatcherWithTyping(params.dispatcherOptions);
    const result = await dispatchInboundMessage({
        ctx: params.ctx,
        cfg: params.cfg,
        dispatcher,
        replyResolver: params.replyResolver,
        replyOptions: {
            ...params.replyOptions,
            ...replyOptions,
        },
    });
    markDispatchIdle();
    return result;
}
export async function dispatchInboundMessageWithDispatcher(params) {
    const dispatcher = createReplyDispatcher(params.dispatcherOptions);
    const result = await dispatchInboundMessage({
        ctx: params.ctx,
        cfg: params.cfg,
        dispatcher,
        replyResolver: params.replyResolver,
        replyOptions: params.replyOptions,
    });
    await dispatcher.waitForIdle();
    return result;
}
