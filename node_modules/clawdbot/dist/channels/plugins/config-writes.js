import { normalizeAccountId } from "../../routing/session-key.js";
function resolveAccountConfig(accounts, accountId) {
    if (!accounts || typeof accounts !== "object")
        return undefined;
    if (accountId in accounts)
        return accounts[accountId];
    const matchKey = Object.keys(accounts).find((key) => key.toLowerCase() === accountId.toLowerCase());
    return matchKey ? accounts[matchKey] : undefined;
}
export function resolveChannelConfigWrites(params) {
    if (!params.channelId)
        return true;
    const channels = params.cfg.channels;
    const channelConfig = channels?.[params.channelId];
    if (!channelConfig)
        return true;
    const accountId = normalizeAccountId(params.accountId);
    const accountConfig = resolveAccountConfig(channelConfig.accounts, accountId);
    const value = accountConfig?.configWrites ?? channelConfig.configWrites;
    return value !== false;
}
