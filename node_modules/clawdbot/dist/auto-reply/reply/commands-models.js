import { loadModelCatalog } from "../../agents/model-catalog.js";
import { buildAllowedModelSet, buildModelAliasIndex, normalizeProviderId, resolveConfiguredModelRef, resolveModelRefFromString, } from "../../agents/model-selection.js";
import { DEFAULT_MODEL, DEFAULT_PROVIDER } from "../../agents/defaults.js";
const PAGE_SIZE_DEFAULT = 20;
const PAGE_SIZE_MAX = 100;
function formatProviderLine(params) {
    return `- ${params.provider} (${params.count})`;
}
function parseModelsArgs(raw) {
    const trimmed = raw.trim();
    if (!trimmed) {
        return { page: 1, pageSize: PAGE_SIZE_DEFAULT, all: false };
    }
    const tokens = trimmed.split(/\s+/g).filter(Boolean);
    const provider = tokens[0]?.trim();
    let page = 1;
    let all = false;
    for (const token of tokens.slice(1)) {
        const lower = token.toLowerCase();
        if (lower === "all" || lower === "--all") {
            all = true;
            continue;
        }
        if (lower.startsWith("page=")) {
            const value = Number.parseInt(lower.slice("page=".length), 10);
            if (Number.isFinite(value) && value > 0)
                page = value;
            continue;
        }
        if (/^[0-9]+$/.test(lower)) {
            const value = Number.parseInt(lower, 10);
            if (Number.isFinite(value) && value > 0)
                page = value;
        }
    }
    let pageSize = PAGE_SIZE_DEFAULT;
    for (const token of tokens) {
        const lower = token.toLowerCase();
        if (lower.startsWith("limit=") || lower.startsWith("size=")) {
            const rawValue = lower.slice(lower.indexOf("=") + 1);
            const value = Number.parseInt(rawValue, 10);
            if (Number.isFinite(value) && value > 0)
                pageSize = Math.min(PAGE_SIZE_MAX, value);
        }
    }
    return {
        provider: provider ? normalizeProviderId(provider) : undefined,
        page,
        pageSize,
        all,
    };
}
export async function resolveModelsCommandReply(params) {
    const body = params.commandBodyNormalized.trim();
    if (!body.startsWith("/models"))
        return null;
    const argText = body.replace(/^\/models\b/i, "").trim();
    const { provider, page, pageSize, all } = parseModelsArgs(argText);
    const resolvedDefault = resolveConfiguredModelRef({
        cfg: params.cfg,
        defaultProvider: DEFAULT_PROVIDER,
        defaultModel: DEFAULT_MODEL,
    });
    const catalog = await loadModelCatalog({ config: params.cfg });
    const allowed = buildAllowedModelSet({
        cfg: params.cfg,
        catalog,
        defaultProvider: resolvedDefault.provider,
        defaultModel: resolvedDefault.model,
    });
    const aliasIndex = buildModelAliasIndex({
        cfg: params.cfg,
        defaultProvider: resolvedDefault.provider,
    });
    const byProvider = new Map();
    const add = (p, m) => {
        const key = normalizeProviderId(p);
        const set = byProvider.get(key) ?? new Set();
        set.add(m);
        byProvider.set(key, set);
    };
    const addRawModelRef = (raw) => {
        const trimmed = raw?.trim();
        if (!trimmed)
            return;
        const resolved = resolveModelRefFromString({
            raw: trimmed,
            defaultProvider: resolvedDefault.provider,
            aliasIndex,
        });
        if (!resolved)
            return;
        add(resolved.ref.provider, resolved.ref.model);
    };
    const addModelConfigEntries = () => {
        const modelConfig = params.cfg.agents?.defaults?.model;
        if (typeof modelConfig === "string") {
            addRawModelRef(modelConfig);
        }
        else if (modelConfig && typeof modelConfig === "object") {
            addRawModelRef(modelConfig.primary);
            for (const fallback of modelConfig.fallbacks ?? []) {
                addRawModelRef(fallback);
            }
        }
        const imageConfig = params.cfg.agents?.defaults?.imageModel;
        if (typeof imageConfig === "string") {
            addRawModelRef(imageConfig);
        }
        else if (imageConfig && typeof imageConfig === "object") {
            addRawModelRef(imageConfig.primary);
            for (const fallback of imageConfig.fallbacks ?? []) {
                addRawModelRef(fallback);
            }
        }
    };
    for (const entry of allowed.allowedCatalog) {
        add(entry.provider, entry.id);
    }
    // Include config-only allowlist keys that aren't in the curated catalog.
    for (const raw of Object.keys(params.cfg.agents?.defaults?.models ?? {})) {
        addRawModelRef(raw);
    }
    // Ensure configured defaults/fallbacks/image models show up even when the
    // curated catalog doesn't know about them (custom providers, dev builds, etc.).
    add(resolvedDefault.provider, resolvedDefault.model);
    addModelConfigEntries();
    const providers = [...byProvider.keys()].sort();
    if (!provider) {
        const lines = [
            "Providers:",
            ...providers.map((p) => formatProviderLine({ provider: p, count: byProvider.get(p)?.size ?? 0 })),
            "",
            "Use: /models <provider>",
            "Switch: /model <provider/model>",
        ];
        return { text: lines.join("\n") };
    }
    if (!byProvider.has(provider)) {
        const lines = [
            `Unknown provider: ${provider}`,
            "",
            "Available providers:",
            ...providers.map((p) => `- ${p}`),
            "",
            "Use: /models <provider>",
        ];
        return { text: lines.join("\n") };
    }
    const models = [...(byProvider.get(provider) ?? new Set())].sort();
    const total = models.length;
    if (total === 0) {
        const lines = [
            `Models (${provider}) — none`,
            "",
            "Browse: /models",
            "Switch: /model <provider/model>",
        ];
        return { text: lines.join("\n") };
    }
    const effectivePageSize = all ? total : pageSize;
    const pageCount = effectivePageSize > 0 ? Math.ceil(total / effectivePageSize) : 1;
    const safePage = all ? 1 : Math.max(1, Math.min(page, pageCount));
    if (!all && page !== safePage) {
        const lines = [
            `Page out of range: ${page} (valid: 1-${pageCount})`,
            "",
            `Try: /models ${provider} ${safePage}`,
            `All: /models ${provider} all`,
        ];
        return { text: lines.join("\n") };
    }
    const startIndex = (safePage - 1) * effectivePageSize;
    const endIndexExclusive = Math.min(total, startIndex + effectivePageSize);
    const pageModels = models.slice(startIndex, endIndexExclusive);
    const header = `Models (${provider}) — showing ${startIndex + 1}-${endIndexExclusive} of ${total} (page ${safePage}/${pageCount})`;
    const lines = [header];
    for (const id of pageModels) {
        lines.push(`- ${provider}/${id}`);
    }
    lines.push("", "Switch: /model <provider/model>");
    if (!all && safePage < pageCount) {
        lines.push(`More: /models ${provider} ${safePage + 1}`);
    }
    if (!all) {
        lines.push(`All: /models ${provider} all`);
    }
    const payload = { text: lines.join("\n") };
    return payload;
}
export const handleModelsCommand = async (params, allowTextCommands) => {
    if (!allowTextCommands)
        return null;
    const reply = await resolveModelsCommandReply({
        cfg: params.cfg,
        commandBodyNormalized: params.command.commandBodyNormalized,
    });
    if (!reply)
        return null;
    return { reply, shouldContinue: false };
};
