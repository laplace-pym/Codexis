import { DEFAULT_ACCOUNT_ID, normalizeAccountId, } from "../../routing/session-key.js";
function providerHasAccounts(cfg, providerKey) {
    const base = cfg[providerKey];
    return Boolean(base?.accounts && Object.keys(base.accounts).length > 0);
}
function shouldStoreNameInAccounts(params) {
    if (params.alwaysUseAccounts)
        return true;
    if (params.accountId !== DEFAULT_ACCOUNT_ID)
        return true;
    return providerHasAccounts(params.cfg, params.providerKey);
}
export function applyAccountNameToProviderSection(params) {
    const trimmed = params.name?.trim();
    if (!trimmed)
        return params.cfg;
    const accountId = normalizeAccountId(params.accountId);
    const baseConfig = params.cfg[params.providerKey];
    const base = typeof baseConfig === "object" && baseConfig
        ? baseConfig
        : undefined;
    const useAccounts = shouldStoreNameInAccounts({
        cfg: params.cfg,
        providerKey: params.providerKey,
        accountId,
        alwaysUseAccounts: params.alwaysUseAccounts,
    });
    if (!useAccounts && accountId === DEFAULT_ACCOUNT_ID) {
        const safeBase = base ?? {};
        return {
            ...params.cfg,
            [params.providerKey]: {
                ...safeBase,
                name: trimmed,
            },
        };
    }
    const baseAccounts = base?.accounts ?? {};
    const existingAccount = baseAccounts[accountId] ?? {};
    const baseWithoutName = accountId === DEFAULT_ACCOUNT_ID
        ? (({ name: _ignored, ...rest }) => rest)(base ?? {})
        : (base ?? {});
    return {
        ...params.cfg,
        [params.providerKey]: {
            ...baseWithoutName,
            accounts: {
                ...baseAccounts,
                [accountId]: {
                    ...existingAccount,
                    name: trimmed,
                },
            },
        },
    };
}
export function migrateBaseNameToDefaultAccount(params) {
    if (params.alwaysUseAccounts)
        return params.cfg;
    const base = params.cfg[params.providerKey];
    const baseName = base?.name?.trim();
    if (!baseName)
        return params.cfg;
    const accounts = {
        ...base?.accounts,
    };
    const defaultAccount = accounts[DEFAULT_ACCOUNT_ID] ?? {};
    if (!defaultAccount.name) {
        accounts[DEFAULT_ACCOUNT_ID] = { ...defaultAccount, name: baseName };
    }
    const { name: _ignored, ...rest } = base ?? {};
    return {
        ...params.cfg,
        [params.providerKey]: {
            ...rest,
            accounts,
        },
    };
}
