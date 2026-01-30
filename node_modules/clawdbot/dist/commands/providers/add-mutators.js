import { getProviderPlugin } from "../../providers/plugins/index.js";
import { normalizeAccountId } from "../../routing/session-key.js";
export function applyAccountName(params) {
    const accountId = normalizeAccountId(params.accountId);
    const plugin = getProviderPlugin(params.provider);
    const apply = plugin?.setup?.applyAccountName;
    return apply
        ? apply({ cfg: params.cfg, accountId, name: params.name })
        : params.cfg;
}
export function applyProviderAccountConfig(params) {
    const accountId = normalizeAccountId(params.accountId);
    const plugin = getProviderPlugin(params.provider);
    const apply = plugin?.setup?.applyAccountConfig;
    if (!apply)
        return params.cfg;
    const input = {
        name: params.name,
        token: params.token,
        tokenFile: params.tokenFile,
        botToken: params.botToken,
        appToken: params.appToken,
        signalNumber: params.signalNumber,
        cliPath: params.cliPath,
        dbPath: params.dbPath,
        service: params.service,
        region: params.region,
        authDir: params.authDir,
        httpUrl: params.httpUrl,
        httpHost: params.httpHost,
        httpPort: params.httpPort,
        useEnv: params.useEnv,
    };
    return apply({ cfg: params.cfg, accountId, input });
}
