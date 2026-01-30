export const CONTEXT_WINDOW_HARD_MIN_TOKENS = 16_000;
export const CONTEXT_WINDOW_WARN_BELOW_TOKENS = 32_000;
function normalizePositiveInt(value) {
    if (typeof value !== "number" || !Number.isFinite(value))
        return null;
    const int = Math.floor(value);
    return int > 0 ? int : null;
}
export function resolveContextWindowInfo(params) {
    const fromModel = normalizePositiveInt(params.modelContextWindow);
    if (fromModel)
        return { tokens: fromModel, source: "model" };
    const fromModelsConfig = (() => {
        const providers = params.cfg?.models?.providers;
        const providerEntry = providers?.[params.provider];
        const models = Array.isArray(providerEntry?.models) ? providerEntry.models : [];
        const match = models.find((m) => m?.id === params.modelId);
        return normalizePositiveInt(match?.contextWindow);
    })();
    if (fromModelsConfig)
        return { tokens: fromModelsConfig, source: "modelsConfig" };
    const fromAgentConfig = normalizePositiveInt(params.cfg?.agents?.defaults?.contextTokens);
    if (fromAgentConfig)
        return { tokens: fromAgentConfig, source: "agentContextTokens" };
    return { tokens: Math.floor(params.defaultTokens), source: "default" };
}
export function evaluateContextWindowGuard(params) {
    const warnBelow = Math.max(1, Math.floor(params.warnBelowTokens ?? CONTEXT_WINDOW_WARN_BELOW_TOKENS));
    const hardMin = Math.max(1, Math.floor(params.hardMinTokens ?? CONTEXT_WINDOW_HARD_MIN_TOKENS));
    const tokens = Math.max(0, Math.floor(params.info.tokens));
    return {
        ...params.info,
        tokens,
        shouldWarn: tokens > 0 && tokens < warnBelow,
        shouldBlock: tokens > 0 && tokens < hardMin,
    };
}
