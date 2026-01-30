import { Bot } from "grammy";
import { resolveTelegramFetch } from "./fetch.js";
export async function setTelegramWebhook(opts) {
    const fetchImpl = resolveTelegramFetch();
    const client = fetchImpl
        ? { fetch: fetchImpl }
        : undefined;
    const bot = new Bot(opts.token, client ? { client } : undefined);
    await bot.api.setWebhook(opts.url, {
        secret_token: opts.secret,
        drop_pending_updates: opts.dropPendingUpdates ?? false,
    });
}
export async function deleteTelegramWebhook(opts) {
    const fetchImpl = resolveTelegramFetch();
    const client = fetchImpl
        ? { fetch: fetchImpl }
        : undefined;
    const bot = new Bot(opts.token, client ? { client } : undefined);
    await bot.api.deleteWebhook();
}
