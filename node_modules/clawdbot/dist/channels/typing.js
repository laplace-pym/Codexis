export function createTypingCallbacks(params) {
    const stop = params.stop;
    const onReplyStart = async () => {
        try {
            await params.start();
        }
        catch (err) {
            params.onStartError(err);
        }
    };
    const onIdle = stop
        ? () => {
            void stop().catch((err) => (params.onStopError ?? params.onStartError)(err));
        }
        : undefined;
    return { onReplyStart, onIdle };
}
