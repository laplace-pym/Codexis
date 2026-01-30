import { normalizeAccountId } from "../../routing/session-key.js";
const MB = 1024 * 1024;
export function resolveProviderMediaMaxBytes(params) {
    const accountId = normalizeAccountId(params.accountId);
    const providerLimit = params.resolveProviderLimitMb({
        cfg: params.cfg,
        accountId,
    });
    if (providerLimit)
        return providerLimit * MB;
    if (params.cfg.agents?.defaults?.mediaMaxMb) {
        return params.cfg.agents.defaults.mediaMaxMb * MB;
    }
    return undefined;
}
