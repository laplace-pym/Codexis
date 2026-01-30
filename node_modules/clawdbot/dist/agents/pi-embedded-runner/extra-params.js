import { streamSimple } from "@mariozechner/pi-ai";
import { log } from "./logger.js";
/**
 * Resolve provider-specific extra params from model config.
 * Used to pass through stream params like temperature/maxTokens.
 *
 * @internal Exported for testing only
 */
export function resolveExtraParams(params) {
    const modelKey = `${params.provider}/${params.modelId}`;
    const modelConfig = params.cfg?.agents?.defaults?.models?.[modelKey];
    return modelConfig?.params ? { ...modelConfig.params } : undefined;
}
function resolveCacheControlTtl(extraParams, provider, modelId) {
    const raw = extraParams?.cacheControlTtl;
    if (raw !== "5m" && raw !== "1h")
        return undefined;
    if (provider === "anthropic")
        return raw;
    if (provider === "openrouter" && modelId.startsWith("anthropic/"))
        return raw;
    return undefined;
}
function createStreamFnWithExtraParams(baseStreamFn, extraParams, provider, modelId) {
    if (!extraParams || Object.keys(extraParams).length === 0) {
        return undefined;
    }
    const streamParams = {};
    if (typeof extraParams.temperature === "number") {
        streamParams.temperature = extraParams.temperature;
    }
    if (typeof extraParams.maxTokens === "number") {
        streamParams.maxTokens = extraParams.maxTokens;
    }
    const cacheControlTtl = resolveCacheControlTtl(extraParams, provider, modelId);
    if (cacheControlTtl) {
        streamParams.cacheControlTtl = cacheControlTtl;
    }
    if (Object.keys(streamParams).length === 0) {
        return undefined;
    }
    log.debug(`creating streamFn wrapper with params: ${JSON.stringify(streamParams)}`);
    const underlying = baseStreamFn ?? streamSimple;
    const wrappedStreamFn = (model, context, options) => underlying(model, context, {
        ...streamParams,
        ...options,
    });
    return wrappedStreamFn;
}
/**
 * Apply extra params (like temperature) to an agent's streamFn.
 *
 * @internal Exported for testing
 */
export function applyExtraParamsToAgent(agent, cfg, provider, modelId, extraParamsOverride) {
    const extraParams = resolveExtraParams({
        cfg,
        provider,
        modelId,
    });
    const override = extraParamsOverride && Object.keys(extraParamsOverride).length > 0
        ? Object.fromEntries(Object.entries(extraParamsOverride).filter(([, value]) => value !== undefined))
        : undefined;
    const merged = Object.assign({}, extraParams, override);
    const wrappedStreamFn = createStreamFnWithExtraParams(agent.streamFn, merged, provider, modelId);
    if (wrappedStreamFn) {
        log.debug(`applying extraParams to agent streamFn for ${provider}/${modelId}`);
        agent.streamFn = wrappedStreamFn;
    }
}
