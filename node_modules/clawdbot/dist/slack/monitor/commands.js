export function normalizeSlackSlashCommandName(raw) {
    return raw.replace(/^\/+/, "");
}
export function resolveSlackSlashCommandConfig(raw) {
    const normalizedName = normalizeSlackSlashCommandName(raw?.name?.trim() || "clawd");
    const name = normalizedName || "clawd";
    return {
        enabled: raw?.enabled === true,
        name,
        sessionPrefix: raw?.sessionPrefix?.trim() || "slack:slash",
        ephemeral: raw?.ephemeral !== false,
    };
}
export function buildSlackSlashCommandMatcher(name) {
    const escaped = name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    return new RegExp(`^/?${escaped}$`);
}
