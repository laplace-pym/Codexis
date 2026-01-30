import { run } from "@grammyjs/runner";
import { loadConfig } from "../config/config.js";
import { resolveAgentMaxConcurrent } from "../config/agent-limits.js";
import { computeBackoff, sleepWithAbort } from "../infra/backoff.js";
import { formatDurationMs } from "../infra/format-duration.js";
import { resolveTelegramAccount } from "./accounts.js";
import { resolveTelegramAllowedUpdates } from "./allowed-updates.js";
import { createTelegramBot } from "./bot.js";
import { makeProxyFetch } from "./proxy.js";
import { readTelegramUpdateOffset, writeTelegramUpdateOffset } from "./update-offset-store.js";
import { startTelegramWebhook } from "./webhook.js";
export function createTelegramRunnerOptions(cfg) {
    return {
        sink: {
            concurrency: resolveAgentMaxConcurrent(cfg),
        },
        runner: {
            fetch: {
                // Match grammY defaults
                timeout: 30,
                // Request reactions without dropping default update types.
                allowed_updates: resolveTelegramAllowedUpdates(),
            },
            // Suppress grammY getUpdates stack traces; we log concise errors ourselves.
            silent: true,
        },
    };
}
const TELEGRAM_POLL_RESTART_POLICY = {
    initialMs: 2000,
    maxMs: 30_000,
    factor: 1.8,
    jitter: 0.25,
};
const isGetUpdatesConflict = (err) => {
    if (!err || typeof err !== "object")
        return false;
    const typed = err;
    const errorCode = typed.error_code ?? typed.errorCode;
    if (errorCode !== 409)
        return false;
    const haystack = [typed.method, typed.description, typed.message]
        .filter((value) => typeof value === "string")
        .join(" ")
        .toLowerCase();
    return haystack.includes("getupdates");
};
export async function monitorTelegramProvider(opts = {}) {
    const cfg = opts.config ?? loadConfig();
    const account = resolveTelegramAccount({
        cfg,
        accountId: opts.accountId,
    });
    const token = opts.token?.trim() || account.token;
    if (!token) {
        throw new Error(`Telegram bot token missing for account "${account.accountId}" (set channels.telegram.accounts.${account.accountId}.botToken/tokenFile or TELEGRAM_BOT_TOKEN for default).`);
    }
    const proxyFetch = opts.proxyFetch ??
        (account.config.proxy ? makeProxyFetch(account.config.proxy) : undefined);
    let lastUpdateId = await readTelegramUpdateOffset({
        accountId: account.accountId,
    });
    const persistUpdateId = async (updateId) => {
        if (lastUpdateId !== null && updateId <= lastUpdateId)
            return;
        lastUpdateId = updateId;
        try {
            await writeTelegramUpdateOffset({
                accountId: account.accountId,
                updateId,
            });
        }
        catch (err) {
            (opts.runtime?.error ?? console.error)(`telegram: failed to persist update offset: ${String(err)}`);
        }
    };
    const bot = createTelegramBot({
        token,
        runtime: opts.runtime,
        proxyFetch,
        config: cfg,
        accountId: account.accountId,
        updateOffset: {
            lastUpdateId,
            onUpdateId: persistUpdateId,
        },
    });
    if (opts.useWebhook) {
        await startTelegramWebhook({
            token,
            accountId: account.accountId,
            config: cfg,
            path: opts.webhookPath,
            port: opts.webhookPort,
            secret: opts.webhookSecret,
            runtime: opts.runtime,
            fetch: proxyFetch,
            abortSignal: opts.abortSignal,
            publicUrl: opts.webhookUrl,
        });
        return;
    }
    // Use grammyjs/runner for concurrent update processing
    const log = opts.runtime?.log ?? console.log;
    let restartAttempts = 0;
    while (!opts.abortSignal?.aborted) {
        const runner = run(bot, createTelegramRunnerOptions(cfg));
        const stopOnAbort = () => {
            if (opts.abortSignal?.aborted) {
                void runner.stop();
            }
        };
        opts.abortSignal?.addEventListener("abort", stopOnAbort, { once: true });
        try {
            // runner.task() returns a promise that resolves when the runner stops
            await runner.task();
            return;
        }
        catch (err) {
            if (opts.abortSignal?.aborted) {
                throw err;
            }
            if (!isGetUpdatesConflict(err)) {
                throw err;
            }
            restartAttempts += 1;
            const delayMs = computeBackoff(TELEGRAM_POLL_RESTART_POLICY, restartAttempts);
            log(`Telegram getUpdates conflict; retrying in ${formatDurationMs(delayMs)}.`);
            try {
                await sleepWithAbort(delayMs, opts.abortSignal);
            }
            catch (sleepErr) {
                if (opts.abortSignal?.aborted)
                    return;
                throw sleepErr;
            }
        }
        finally {
            opts.abortSignal?.removeEventListener("abort", stopOnAbort);
        }
    }
}
