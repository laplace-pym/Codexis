import { messagingApi } from "@line/bot-sdk";
export async function probeLineBot(channelAccessToken, timeoutMs = 5000) {
    if (!channelAccessToken?.trim()) {
        return { ok: false, error: "Channel access token not configured" };
    }
    const client = new messagingApi.MessagingApiClient({
        channelAccessToken: channelAccessToken.trim(),
    });
    try {
        const profile = await withTimeout(client.getBotInfo(), timeoutMs);
        return {
            ok: true,
            bot: {
                displayName: profile.displayName,
                userId: profile.userId,
                basicId: profile.basicId,
                pictureUrl: profile.pictureUrl,
            },
        };
    }
    catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        return { ok: false, error: message };
    }
}
function withTimeout(promise, timeoutMs) {
    if (!timeoutMs || timeoutMs <= 0)
        return promise;
    let timer = null;
    const timeout = new Promise((_, reject) => {
        timer = setTimeout(() => reject(new Error("timeout")), timeoutMs);
    });
    return Promise.race([promise, timeout]).finally(() => {
        if (timer)
            clearTimeout(timer);
    });
}
