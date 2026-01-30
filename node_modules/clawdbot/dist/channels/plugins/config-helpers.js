import { DEFAULT_ACCOUNT_ID } from "../../routing/session-key.js";
export function setAccountEnabledInConfigSection(params) {
    const accountKey = params.accountId || DEFAULT_ACCOUNT_ID;
    const channels = params.cfg.channels;
    const base = channels?.[params.sectionKey];
    const hasAccounts = Boolean(base?.accounts);
    if (params.allowTopLevel && accountKey === DEFAULT_ACCOUNT_ID && !hasAccounts) {
        return {
            ...params.cfg,
            channels: {
                ...params.cfg.channels,
                [params.sectionKey]: {
                    ...base,
                    enabled: params.enabled,
                },
            },
        };
    }
    const baseAccounts = (base?.accounts ?? {});
    const existing = baseAccounts[accountKey] ?? {};
    return {
        ...params.cfg,
        channels: {
            ...params.cfg.channels,
            [params.sectionKey]: {
                ...base,
                accounts: {
                    ...baseAccounts,
                    [accountKey]: {
                        ...existing,
                        enabled: params.enabled,
                    },
                },
            },
        },
    };
}
export function deleteAccountFromConfigSection(params) {
    const accountKey = params.accountId || DEFAULT_ACCOUNT_ID;
    const channels = params.cfg.channels;
    const base = channels?.[params.sectionKey];
    if (!base)
        return params.cfg;
    const baseAccounts = base.accounts && typeof base.accounts === "object" ? { ...base.accounts } : undefined;
    if (accountKey !== DEFAULT_ACCOUNT_ID) {
        const accounts = baseAccounts ? { ...baseAccounts } : {};
        delete accounts[accountKey];
        return {
            ...params.cfg,
            channels: {
                ...params.cfg.channels,
                [params.sectionKey]: {
                    ...base,
                    accounts: Object.keys(accounts).length ? accounts : undefined,
                },
            },
        };
    }
    if (baseAccounts && Object.keys(baseAccounts).length > 0) {
        delete baseAccounts[accountKey];
        const baseRecord = { ...base };
        for (const field of params.clearBaseFields ?? []) {
            if (field in baseRecord)
                baseRecord[field] = undefined;
        }
        return {
            ...params.cfg,
            channels: {
                ...params.cfg.channels,
                [params.sectionKey]: {
                    ...baseRecord,
                    accounts: Object.keys(baseAccounts).length ? baseAccounts : undefined,
                },
            },
        };
    }
    const nextChannels = { ...params.cfg.channels };
    delete nextChannels[params.sectionKey];
    const nextCfg = { ...params.cfg };
    if (Object.keys(nextChannels).length > 0) {
        nextCfg.channels = nextChannels;
    }
    else {
        delete nextCfg.channels;
    }
    return nextCfg;
}
