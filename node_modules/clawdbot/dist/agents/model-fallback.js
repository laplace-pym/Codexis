import { DEFAULT_MODEL, DEFAULT_PROVIDER } from "./defaults.js";
import { coerceToFailoverError, describeFailoverError, isFailoverError, isTimeoutError, } from "./failover-error.js";
import { buildModelAliasIndex, modelKey, parseModelRef, resolveConfiguredModelRef, resolveModelRefFromString, } from "./model-selection.js";
function isAbortError(err) {
    if (!err || typeof err !== "object")
        return false;
    if (isFailoverError(err))
        return false;
    const name = "name" in err ? String(err.name) : "";
    // Only treat explicit AbortError names as user aborts.
    // Message-based checks (e.g., "aborted") can mask timeouts and skip fallback.
    return name === "AbortError";
}
function shouldRethrowAbort(err) {
    return isAbortError(err) && !isTimeoutError(err);
}
function buildAllowedModelKeys(cfg, defaultProvider) {
    const rawAllowlist = (() => {
        const modelMap = cfg?.agents?.defaults?.models ?? {};
        return Object.keys(modelMap);
    })();
    if (rawAllowlist.length === 0)
        return null;
    const keys = new Set();
    for (const raw of rawAllowlist) {
        const parsed = parseModelRef(String(raw ?? ""), defaultProvider);
        if (!parsed)
            continue;
        keys.add(modelKey(parsed.provider, parsed.model));
    }
    return keys.size > 0 ? keys : null;
}
function resolveImageFallbackCandidates(params) {
    const aliasIndex = buildModelAliasIndex({
        cfg: params.cfg ?? {},
        defaultProvider: params.defaultProvider,
    });
    const allowlist = buildAllowedModelKeys(params.cfg, params.defaultProvider);
    const seen = new Set();
    const candidates = [];
    const addCandidate = (candidate, enforceAllowlist) => {
        if (!candidate.provider || !candidate.model)
            return;
        const key = modelKey(candidate.provider, candidate.model);
        if (seen.has(key))
            return;
        if (enforceAllowlist && allowlist && !allowlist.has(key))
            return;
        seen.add(key);
        candidates.push(candidate);
    };
    const addRaw = (raw, enforceAllowlist) => {
        const resolved = resolveModelRefFromString({
            raw: String(raw ?? ""),
            defaultProvider: params.defaultProvider,
            aliasIndex,
        });
        if (!resolved)
            return;
        addCandidate(resolved.ref, enforceAllowlist);
    };
    if (params.modelOverride?.trim()) {
        addRaw(params.modelOverride, false);
    }
    else {
        const imageModel = params.cfg?.agents?.defaults?.imageModel;
        const primary = typeof imageModel === "string" ? imageModel.trim() : imageModel?.primary;
        if (primary?.trim())
            addRaw(primary, false);
    }
    const imageFallbacks = (() => {
        const imageModel = params.cfg?.agents?.defaults?.imageModel;
        if (imageModel && typeof imageModel === "object") {
            return imageModel.fallbacks ?? [];
        }
        return [];
    })();
    for (const raw of imageFallbacks) {
        addRaw(raw, true);
    }
    return candidates;
}
function resolveFallbackCandidates(params) {
    const primary = params.cfg
        ? resolveConfiguredModelRef({
            cfg: params.cfg,
            defaultProvider: DEFAULT_PROVIDER,
            defaultModel: DEFAULT_MODEL,
        })
        : null;
    const defaultProvider = primary?.provider ?? DEFAULT_PROVIDER;
    const defaultModel = primary?.model ?? DEFAULT_MODEL;
    const provider = String(params.provider ?? "").trim() || defaultProvider;
    const model = String(params.model ?? "").trim() || defaultModel;
    const aliasIndex = buildModelAliasIndex({
        cfg: params.cfg ?? {},
        defaultProvider,
    });
    const allowlist = buildAllowedModelKeys(params.cfg, defaultProvider);
    const seen = new Set();
    const candidates = [];
    const addCandidate = (candidate, enforceAllowlist) => {
        if (!candidate.provider || !candidate.model)
            return;
        const key = modelKey(candidate.provider, candidate.model);
        if (seen.has(key))
            return;
        if (enforceAllowlist && allowlist && !allowlist.has(key))
            return;
        seen.add(key);
        candidates.push(candidate);
    };
    addCandidate({ provider, model }, false);
    const modelFallbacks = (() => {
        if (params.fallbacksOverride !== undefined)
            return params.fallbacksOverride;
        const model = params.cfg?.agents?.defaults?.model;
        if (model && typeof model === "object")
            return model.fallbacks ?? [];
        return [];
    })();
    for (const raw of modelFallbacks) {
        const resolved = resolveModelRefFromString({
            raw: String(raw ?? ""),
            defaultProvider,
            aliasIndex,
        });
        if (!resolved)
            continue;
        addCandidate(resolved.ref, true);
    }
    if (params.fallbacksOverride === undefined && primary?.provider && primary.model) {
        addCandidate({ provider: primary.provider, model: primary.model }, false);
    }
    return candidates;
}
export async function runWithModelFallback(params) {
    const candidates = resolveFallbackCandidates({
        cfg: params.cfg,
        provider: params.provider,
        model: params.model,
        fallbacksOverride: params.fallbacksOverride,
    });
    const attempts = [];
    let lastError;
    for (let i = 0; i < candidates.length; i += 1) {
        const candidate = candidates[i];
        try {
            const result = await params.run(candidate.provider, candidate.model);
            return {
                result,
                provider: candidate.provider,
                model: candidate.model,
                attempts,
            };
        }
        catch (err) {
            if (shouldRethrowAbort(err))
                throw err;
            const normalized = coerceToFailoverError(err, {
                provider: candidate.provider,
                model: candidate.model,
            }) ?? err;
            if (!isFailoverError(normalized))
                throw err;
            lastError = normalized;
            const described = describeFailoverError(normalized);
            attempts.push({
                provider: candidate.provider,
                model: candidate.model,
                error: described.message,
                reason: described.reason,
                status: described.status,
                code: described.code,
            });
            await params.onError?.({
                provider: candidate.provider,
                model: candidate.model,
                error: normalized,
                attempt: i + 1,
                total: candidates.length,
            });
        }
    }
    if (attempts.length <= 1 && lastError)
        throw lastError;
    const summary = attempts.length > 0
        ? attempts
            .map((attempt) => `${attempt.provider}/${attempt.model}: ${attempt.error}${attempt.reason ? ` (${attempt.reason})` : ""}`)
            .join(" | ")
        : "unknown";
    throw new Error(`All models failed (${attempts.length || candidates.length}): ${summary}`, {
        cause: lastError instanceof Error ? lastError : undefined,
    });
}
export async function runWithImageModelFallback(params) {
    const candidates = resolveImageFallbackCandidates({
        cfg: params.cfg,
        defaultProvider: DEFAULT_PROVIDER,
        modelOverride: params.modelOverride,
    });
    if (candidates.length === 0) {
        throw new Error("No image model configured. Set agents.defaults.imageModel.primary or agents.defaults.imageModel.fallbacks.");
    }
    const attempts = [];
    let lastError;
    for (let i = 0; i < candidates.length; i += 1) {
        const candidate = candidates[i];
        try {
            const result = await params.run(candidate.provider, candidate.model);
            return {
                result,
                provider: candidate.provider,
                model: candidate.model,
                attempts,
            };
        }
        catch (err) {
            if (shouldRethrowAbort(err))
                throw err;
            lastError = err;
            attempts.push({
                provider: candidate.provider,
                model: candidate.model,
                error: err instanceof Error ? err.message : String(err),
            });
            await params.onError?.({
                provider: candidate.provider,
                model: candidate.model,
                error: err,
                attempt: i + 1,
                total: candidates.length,
            });
        }
    }
    if (attempts.length <= 1 && lastError)
        throw lastError;
    const summary = attempts.length > 0
        ? attempts
            .map((attempt) => `${attempt.provider}/${attempt.model}: ${attempt.error}`)
            .join(" | ")
        : "unknown";
    throw new Error(`All image models failed (${attempts.length || candidates.length}): ${summary}`, {
        cause: lastError instanceof Error ? lastError : undefined,
    });
}
