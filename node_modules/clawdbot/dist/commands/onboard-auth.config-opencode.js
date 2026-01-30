import { OPENCODE_ZEN_DEFAULT_MODEL_REF } from "../agents/opencode-zen-models.js";
export function applyOpencodeZenProviderConfig(cfg) {
    // Use the built-in opencode provider from pi-ai; only seed the allowlist alias.
    const models = { ...cfg.agents?.defaults?.models };
    models[OPENCODE_ZEN_DEFAULT_MODEL_REF] = {
        ...models[OPENCODE_ZEN_DEFAULT_MODEL_REF],
        alias: models[OPENCODE_ZEN_DEFAULT_MODEL_REF]?.alias ?? "Opus",
    };
    return {
        ...cfg,
        agents: {
            ...cfg.agents,
            defaults: {
                ...cfg.agents?.defaults,
                models,
            },
        },
    };
}
export function applyOpencodeZenConfig(cfg) {
    const next = applyOpencodeZenProviderConfig(cfg);
    return {
        ...next,
        agents: {
            ...next.agents,
            defaults: {
                ...next.agents?.defaults,
                model: {
                    ...(next.agents?.defaults?.model &&
                        "fallbacks" in next.agents.defaults.model
                        ? {
                            fallbacks: next.agents.defaults.model.fallbacks,
                        }
                        : undefined),
                    primary: OPENCODE_ZEN_DEFAULT_MODEL_REF,
                },
            },
        },
    };
}
