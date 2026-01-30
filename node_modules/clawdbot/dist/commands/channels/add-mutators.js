import { getChannelPlugin } from "../../channels/plugins/index.js";
import { normalizeAccountId } from "../../routing/session-key.js";
export function applyAccountName(params) {
    const accountId = normalizeAccountId(params.accountId);
    const plugin = getChannelPlugin(params.channel);
    const apply = plugin?.setup?.applyAccountName;
    return apply ? apply({ cfg: params.cfg, accountId, name: params.name }) : params.cfg;
}
export function applyChannelAccountConfig(params) {
    const accountId = normalizeAccountId(params.accountId);
    const plugin = getChannelPlugin(params.channel);
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
        webhookPath: params.webhookPath,
        webhookUrl: params.webhookUrl,
        audienceType: params.audienceType,
        audience: params.audience,
        useEnv: params.useEnv,
        homeserver: params.homeserver,
        userId: params.userId,
        accessToken: params.accessToken,
        password: params.password,
        deviceName: params.deviceName,
        initialSyncLimit: params.initialSyncLimit,
        ship: params.ship,
        url: params.url,
        code: params.code,
        groupChannels: params.groupChannels,
        dmAllowlist: params.dmAllowlist,
        autoDiscoverChannels: params.autoDiscoverChannels,
    };
    return apply({ cfg: params.cfg, accountId, input });
}
