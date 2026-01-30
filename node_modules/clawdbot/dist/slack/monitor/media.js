import { fetchRemoteMedia } from "../../media/fetch.js";
import { saveMediaBuffer } from "../../media/store.js";
export async function resolveSlackMedia(params) {
    const files = params.files ?? [];
    for (const file of files) {
        const url = file.url_private_download ?? file.url_private;
        if (!url)
            continue;
        try {
            const fetchImpl = (input, init) => {
                const headers = new Headers(init?.headers);
                headers.set("Authorization", `Bearer ${params.token}`);
                return fetch(input, { ...init, headers });
            };
            const fetched = await fetchRemoteMedia({
                url,
                fetchImpl,
                filePathHint: file.name,
            });
            if (fetched.buffer.byteLength > params.maxBytes)
                continue;
            const saved = await saveMediaBuffer(fetched.buffer, fetched.contentType ?? file.mimetype, "inbound", params.maxBytes);
            const label = fetched.fileName ?? file.name;
            return {
                path: saved.path,
                contentType: saved.contentType,
                placeholder: label ? `[Slack file: ${label}]` : "[Slack file]",
            };
        }
        catch {
            // Ignore download failures and fall through to the next file.
        }
    }
    return null;
}
const THREAD_STARTER_CACHE = new Map();
export async function resolveSlackThreadStarter(params) {
    const cacheKey = `${params.channelId}:${params.threadTs}`;
    const cached = THREAD_STARTER_CACHE.get(cacheKey);
    if (cached)
        return cached;
    try {
        const response = (await params.client.conversations.replies({
            channel: params.channelId,
            ts: params.threadTs,
            limit: 1,
            inclusive: true,
        }));
        const message = response?.messages?.[0];
        const text = (message?.text ?? "").trim();
        if (!message || !text)
            return null;
        const starter = {
            text,
            userId: message.user,
            ts: message.ts,
            files: message.files,
        };
        THREAD_STARTER_CACHE.set(cacheKey, starter);
        return starter;
    }
    catch {
        return null;
    }
}
