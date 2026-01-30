import { setCliSessionId } from "../../agents/cli-session.js";
import { hasNonzeroUsage } from "../../agents/usage.js";
import { updateSessionStoreEntry, } from "../../config/sessions.js";
import { logVerbose } from "../../globals.js";
export async function persistSessionUsageUpdate(params) {
    const { storePath, sessionKey } = params;
    if (!storePath || !sessionKey)
        return;
    const label = params.logLabel ? `${params.logLabel} ` : "";
    if (hasNonzeroUsage(params.usage)) {
        try {
            await updateSessionStoreEntry({
                storePath,
                sessionKey,
                update: async (entry) => {
                    const input = params.usage?.input ?? 0;
                    const output = params.usage?.output ?? 0;
                    const promptTokens = input + (params.usage?.cacheRead ?? 0) + (params.usage?.cacheWrite ?? 0);
                    const patch = {
                        inputTokens: input,
                        outputTokens: output,
                        totalTokens: promptTokens > 0 ? promptTokens : (params.usage?.total ?? input),
                        modelProvider: params.providerUsed ?? entry.modelProvider,
                        model: params.modelUsed ?? entry.model,
                        contextTokens: params.contextTokensUsed ?? entry.contextTokens,
                        systemPromptReport: params.systemPromptReport ?? entry.systemPromptReport,
                        updatedAt: Date.now(),
                    };
                    const cliProvider = params.providerUsed ?? entry.modelProvider;
                    if (params.cliSessionId && cliProvider) {
                        const nextEntry = { ...entry, ...patch };
                        setCliSessionId(nextEntry, cliProvider, params.cliSessionId);
                        return {
                            ...patch,
                            cliSessionIds: nextEntry.cliSessionIds,
                            claudeCliSessionId: nextEntry.claudeCliSessionId,
                        };
                    }
                    return patch;
                },
            });
        }
        catch (err) {
            logVerbose(`failed to persist ${label}usage update: ${String(err)}`);
        }
        return;
    }
    if (params.modelUsed || params.contextTokensUsed) {
        try {
            await updateSessionStoreEntry({
                storePath,
                sessionKey,
                update: async (entry) => {
                    const patch = {
                        modelProvider: params.providerUsed ?? entry.modelProvider,
                        model: params.modelUsed ?? entry.model,
                        contextTokens: params.contextTokensUsed ?? entry.contextTokens,
                        systemPromptReport: params.systemPromptReport ?? entry.systemPromptReport,
                        updatedAt: Date.now(),
                    };
                    const cliProvider = params.providerUsed ?? entry.modelProvider;
                    if (params.cliSessionId && cliProvider) {
                        const nextEntry = { ...entry, ...patch };
                        setCliSessionId(nextEntry, cliProvider, params.cliSessionId);
                        return {
                            ...patch,
                            cliSessionIds: nextEntry.cliSessionIds,
                            claudeCliSessionId: nextEntry.claudeCliSessionId,
                        };
                    }
                    return patch;
                },
            });
        }
        catch (err) {
            logVerbose(`failed to persist ${label}model/context update: ${String(err)}`);
        }
    }
}
