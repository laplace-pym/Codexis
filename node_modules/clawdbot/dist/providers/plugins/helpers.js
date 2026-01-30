import { DEFAULT_ACCOUNT_ID } from "../../routing/session-key.js";
// Provider docking helper: use this when selecting the default account for a plugin.
export function resolveProviderDefaultAccountId(params) {
    const accountIds = params.accountIds ?? params.plugin.config.listAccountIds(params.cfg);
    return (params.plugin.config.defaultAccountId?.(params.cfg) ??
        accountIds[0] ??
        DEFAULT_ACCOUNT_ID);
}
export function formatPairingApproveHint(providerId) {
    return `Approve via: clawdbot pairing list ${providerId} / clawdbot pairing approve ${providerId} <code>`;
}
