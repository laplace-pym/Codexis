import { normalizeAccountId } from "../routing/session-key.js";
function resolveAccountGroups(cfg, accountId) {
    if (!accountId)
        return {};
    const normalized = normalizeAccountId(accountId);
    const accounts = cfg.channels?.telegram?.accounts;
    if (!accounts || typeof accounts !== "object")
        return {};
    const exact = accounts[normalized];
    if (exact?.groups)
        return { groups: exact.groups };
    const matchKey = Object.keys(accounts).find((key) => key.toLowerCase() === normalized.toLowerCase());
    return { groups: matchKey ? accounts[matchKey]?.groups : undefined };
}
export function migrateTelegramGroupsInPlace(groups, oldChatId, newChatId) {
    if (!groups)
        return { migrated: false, skippedExisting: false };
    if (oldChatId === newChatId)
        return { migrated: false, skippedExisting: false };
    if (!Object.hasOwn(groups, oldChatId))
        return { migrated: false, skippedExisting: false };
    if (Object.hasOwn(groups, newChatId))
        return { migrated: false, skippedExisting: true };
    groups[newChatId] = groups[oldChatId];
    delete groups[oldChatId];
    return { migrated: true, skippedExisting: false };
}
export function migrateTelegramGroupConfig(params) {
    const scopes = [];
    let migrated = false;
    let skippedExisting = false;
    const accountGroups = resolveAccountGroups(params.cfg, params.accountId).groups;
    if (accountGroups) {
        const result = migrateTelegramGroupsInPlace(accountGroups, params.oldChatId, params.newChatId);
        if (result.migrated) {
            migrated = true;
            scopes.push("account");
        }
        if (result.skippedExisting)
            skippedExisting = true;
    }
    const globalGroups = params.cfg.channels?.telegram?.groups;
    if (globalGroups) {
        const result = migrateTelegramGroupsInPlace(globalGroups, params.oldChatId, params.newChatId);
        if (result.migrated) {
            migrated = true;
            scopes.push("global");
        }
        if (result.skippedExisting)
            skippedExisting = true;
    }
    return { migrated, skippedExisting, scopes };
}
