import { DEFAULT_ACCOUNT_ID, normalizeAccountId } from "../routing/session-key.js";
function listConfiguredAccountIds(cfg) {
    const accounts = cfg.channels?.signal?.accounts;
    if (!accounts || typeof accounts !== "object")
        return [];
    return Object.keys(accounts).filter(Boolean);
}
export function listSignalAccountIds(cfg) {
    const ids = listConfiguredAccountIds(cfg);
    if (ids.length === 0)
        return [DEFAULT_ACCOUNT_ID];
    return ids.sort((a, b) => a.localeCompare(b));
}
export function resolveDefaultSignalAccountId(cfg) {
    const ids = listSignalAccountIds(cfg);
    if (ids.includes(DEFAULT_ACCOUNT_ID))
        return DEFAULT_ACCOUNT_ID;
    return ids[0] ?? DEFAULT_ACCOUNT_ID;
}
function resolveAccountConfig(cfg, accountId) {
    const accounts = cfg.channels?.signal?.accounts;
    if (!accounts || typeof accounts !== "object")
        return undefined;
    return accounts[accountId];
}
function mergeSignalAccountConfig(cfg, accountId) {
    const { accounts: _ignored, ...base } = (cfg.channels?.signal ?? {});
    const account = resolveAccountConfig(cfg, accountId) ?? {};
    return { ...base, ...account };
}
export function resolveSignalAccount(params) {
    const accountId = normalizeAccountId(params.accountId);
    const baseEnabled = params.cfg.channels?.signal?.enabled !== false;
    const merged = mergeSignalAccountConfig(params.cfg, accountId);
    const accountEnabled = merged.enabled !== false;
    const enabled = baseEnabled && accountEnabled;
    const host = merged.httpHost?.trim() || "127.0.0.1";
    const port = merged.httpPort ?? 8080;
    const baseUrl = merged.httpUrl?.trim() || `http://${host}:${port}`;
    const configured = Boolean(merged.account?.trim() ||
        merged.httpUrl?.trim() ||
        merged.cliPath?.trim() ||
        merged.httpHost?.trim() ||
        typeof merged.httpPort === "number" ||
        typeof merged.autoStart === "boolean");
    return {
        accountId,
        enabled,
        name: merged.name?.trim() || undefined,
        baseUrl,
        configured,
        config: merged,
    };
}
export function listEnabledSignalAccounts(cfg) {
    return listSignalAccountIds(cfg)
        .map((accountId) => resolveSignalAccount({ cfg, accountId }))
        .filter((account) => account.enabled);
}
