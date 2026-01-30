import { discoverAuthStorage, discoverModels } from "@mariozechner/pi-coding-agent";
import { resolveClawdbotAgentDir } from "../agent-paths.js";
import { DEFAULT_CONTEXT_TOKENS } from "../defaults.js";
import { normalizeModelCompat } from "../model-compat.js";
import { normalizeProviderId } from "../model-selection.js";
export function buildInlineProviderModels(providers) {
    return Object.entries(providers).flatMap(([providerId, entry]) => {
        const trimmed = providerId.trim();
        if (!trimmed)
            return [];
        return (entry?.models ?? []).map((model) => ({ ...model, provider: trimmed }));
    });
}
export function buildModelAliasLines(cfg) {
    const models = cfg?.agents?.defaults?.models ?? {};
    const entries = [];
    for (const [keyRaw, entryRaw] of Object.entries(models)) {
        const model = String(keyRaw ?? "").trim();
        if (!model)
            continue;
        const alias = String(entryRaw?.alias ?? "").trim();
        if (!alias)
            continue;
        entries.push({ alias, model });
    }
    return entries
        .sort((a, b) => a.alias.localeCompare(b.alias))
        .map((entry) => `- ${entry.alias}: ${entry.model}`);
}
export function resolveModel(provider, modelId, agentDir, cfg) {
    const resolvedAgentDir = agentDir ?? resolveClawdbotAgentDir();
    const authStorage = discoverAuthStorage(resolvedAgentDir);
    const modelRegistry = discoverModels(authStorage, resolvedAgentDir);
    const model = modelRegistry.find(provider, modelId);
    if (!model) {
        const providers = cfg?.models?.providers ?? {};
        const inlineModels = buildInlineProviderModels(providers);
        const normalizedProvider = normalizeProviderId(provider);
        const inlineMatch = inlineModels.find((entry) => normalizeProviderId(entry.provider) === normalizedProvider && entry.id === modelId);
        if (inlineMatch) {
            const normalized = normalizeModelCompat(inlineMatch);
            return {
                model: normalized,
                authStorage,
                modelRegistry,
            };
        }
        const providerCfg = providers[provider];
        if (providerCfg || modelId.startsWith("mock-")) {
            const fallbackModel = normalizeModelCompat({
                id: modelId,
                name: modelId,
                api: providerCfg?.api ?? "openai-responses",
                provider,
                reasoning: false,
                input: ["text"],
                cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
                contextWindow: providerCfg?.models?.[0]?.contextWindow ?? DEFAULT_CONTEXT_TOKENS,
                maxTokens: providerCfg?.models?.[0]?.maxTokens ?? DEFAULT_CONTEXT_TOKENS,
            });
            return { model: fallbackModel, authStorage, modelRegistry };
        }
        return {
            error: `Unknown model: ${provider}/${modelId}`,
            authStorage,
            modelRegistry,
        };
    }
    return { model: normalizeModelCompat(model), authStorage, modelRegistry };
}
