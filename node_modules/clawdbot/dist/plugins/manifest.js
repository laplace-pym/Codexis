import fs from "node:fs";
import path from "node:path";
export const PLUGIN_MANIFEST_FILENAME = "clawdbot.plugin.json";
function normalizeStringList(value) {
    if (!Array.isArray(value))
        return [];
    return value.map((entry) => (typeof entry === "string" ? entry.trim() : "")).filter(Boolean);
}
function isRecord(value) {
    return Boolean(value && typeof value === "object" && !Array.isArray(value));
}
export function resolvePluginManifestPath(rootDir) {
    return path.join(rootDir, PLUGIN_MANIFEST_FILENAME);
}
export function loadPluginManifest(rootDir) {
    const manifestPath = resolvePluginManifestPath(rootDir);
    if (!fs.existsSync(manifestPath)) {
        return { ok: false, error: `plugin manifest not found: ${manifestPath}`, manifestPath };
    }
    let raw;
    try {
        raw = JSON.parse(fs.readFileSync(manifestPath, "utf-8"));
    }
    catch (err) {
        return {
            ok: false,
            error: `failed to parse plugin manifest: ${String(err)}`,
            manifestPath,
        };
    }
    if (!isRecord(raw)) {
        return { ok: false, error: "plugin manifest must be an object", manifestPath };
    }
    const id = typeof raw.id === "string" ? raw.id.trim() : "";
    if (!id) {
        return { ok: false, error: "plugin manifest requires id", manifestPath };
    }
    const configSchema = isRecord(raw.configSchema) ? raw.configSchema : null;
    if (!configSchema) {
        return { ok: false, error: "plugin manifest requires configSchema", manifestPath };
    }
    const kind = typeof raw.kind === "string" ? raw.kind : undefined;
    const name = typeof raw.name === "string" ? raw.name.trim() : undefined;
    const description = typeof raw.description === "string" ? raw.description.trim() : undefined;
    const version = typeof raw.version === "string" ? raw.version.trim() : undefined;
    const channels = normalizeStringList(raw.channels);
    const providers = normalizeStringList(raw.providers);
    const skills = normalizeStringList(raw.skills);
    let uiHints;
    if (isRecord(raw.uiHints)) {
        uiHints = raw.uiHints;
    }
    return {
        ok: true,
        manifest: {
            id,
            configSchema,
            kind,
            channels,
            providers,
            skills,
            name,
            description,
            version,
            uiHints,
        },
        manifestPath,
    };
}
