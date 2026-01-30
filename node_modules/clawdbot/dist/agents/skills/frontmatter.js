import JSON5 from "json5";
import { parseFrontmatterBlock } from "../../markdown/frontmatter.js";
import { parseBooleanValue } from "../../utils/boolean.js";
export function parseFrontmatter(content) {
    return parseFrontmatterBlock(content);
}
function normalizeStringList(input) {
    if (!input)
        return [];
    if (Array.isArray(input)) {
        return input.map((value) => String(value).trim()).filter(Boolean);
    }
    if (typeof input === "string") {
        return input
            .split(",")
            .map((value) => value.trim())
            .filter(Boolean);
    }
    return [];
}
function parseInstallSpec(input) {
    if (!input || typeof input !== "object")
        return undefined;
    const raw = input;
    const kindRaw = typeof raw.kind === "string" ? raw.kind : typeof raw.type === "string" ? raw.type : "";
    const kind = kindRaw.trim().toLowerCase();
    if (kind !== "brew" && kind !== "node" && kind !== "go" && kind !== "uv" && kind !== "download") {
        return undefined;
    }
    const spec = {
        kind: kind,
    };
    if (typeof raw.id === "string")
        spec.id = raw.id;
    if (typeof raw.label === "string")
        spec.label = raw.label;
    const bins = normalizeStringList(raw.bins);
    if (bins.length > 0)
        spec.bins = bins;
    const osList = normalizeStringList(raw.os);
    if (osList.length > 0)
        spec.os = osList;
    if (typeof raw.formula === "string")
        spec.formula = raw.formula;
    if (typeof raw.package === "string")
        spec.package = raw.package;
    if (typeof raw.module === "string")
        spec.module = raw.module;
    if (typeof raw.url === "string")
        spec.url = raw.url;
    if (typeof raw.archive === "string")
        spec.archive = raw.archive;
    if (typeof raw.extract === "boolean")
        spec.extract = raw.extract;
    if (typeof raw.stripComponents === "number")
        spec.stripComponents = raw.stripComponents;
    if (typeof raw.targetDir === "string")
        spec.targetDir = raw.targetDir;
    return spec;
}
function getFrontmatterValue(frontmatter, key) {
    const raw = frontmatter[key];
    return typeof raw === "string" ? raw : undefined;
}
function parseFrontmatterBool(value, fallback) {
    const parsed = parseBooleanValue(value);
    return parsed === undefined ? fallback : parsed;
}
export function resolveClawdbotMetadata(frontmatter) {
    const raw = getFrontmatterValue(frontmatter, "metadata");
    if (!raw)
        return undefined;
    try {
        const parsed = JSON5.parse(raw);
        if (!parsed || typeof parsed !== "object")
            return undefined;
        const clawdbot = parsed.clawdbot;
        if (!clawdbot || typeof clawdbot !== "object")
            return undefined;
        const clawdbotObj = clawdbot;
        const requiresRaw = typeof clawdbotObj.requires === "object" && clawdbotObj.requires !== null
            ? clawdbotObj.requires
            : undefined;
        const installRaw = Array.isArray(clawdbotObj.install) ? clawdbotObj.install : [];
        const install = installRaw
            .map((entry) => parseInstallSpec(entry))
            .filter((entry) => Boolean(entry));
        const osRaw = normalizeStringList(clawdbotObj.os);
        return {
            always: typeof clawdbotObj.always === "boolean" ? clawdbotObj.always : undefined,
            emoji: typeof clawdbotObj.emoji === "string" ? clawdbotObj.emoji : undefined,
            homepage: typeof clawdbotObj.homepage === "string" ? clawdbotObj.homepage : undefined,
            skillKey: typeof clawdbotObj.skillKey === "string" ? clawdbotObj.skillKey : undefined,
            primaryEnv: typeof clawdbotObj.primaryEnv === "string" ? clawdbotObj.primaryEnv : undefined,
            os: osRaw.length > 0 ? osRaw : undefined,
            requires: requiresRaw
                ? {
                    bins: normalizeStringList(requiresRaw.bins),
                    anyBins: normalizeStringList(requiresRaw.anyBins),
                    env: normalizeStringList(requiresRaw.env),
                    config: normalizeStringList(requiresRaw.config),
                }
                : undefined,
            install: install.length > 0 ? install : undefined,
        };
    }
    catch {
        return undefined;
    }
}
export function resolveSkillInvocationPolicy(frontmatter) {
    return {
        userInvocable: parseFrontmatterBool(getFrontmatterValue(frontmatter, "user-invocable"), true),
        disableModelInvocation: parseFrontmatterBool(getFrontmatterValue(frontmatter, "disable-model-invocation"), false),
    };
}
export function resolveSkillKey(skill, entry) {
    return entry?.clawdbot?.skillKey ?? skill.name;
}
