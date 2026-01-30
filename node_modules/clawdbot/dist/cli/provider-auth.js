import { loadConfig } from "../config/config.js";
import { setVerbose } from "../globals.js";
import { resolveProviderDefaultAccountId } from "../providers/plugins/helpers.js";
import { getProviderPlugin, normalizeProviderId, } from "../providers/plugins/index.js";
import { DEFAULT_CHAT_PROVIDER } from "../providers/registry.js";
import { defaultRuntime } from "../runtime.js";
export async function runProviderLogin(opts, runtime = defaultRuntime) {
    const providerInput = opts.provider ?? DEFAULT_CHAT_PROVIDER;
    const providerId = normalizeProviderId(providerInput);
    if (!providerId) {
        throw new Error(`Unsupported provider: ${providerInput}`);
    }
    const plugin = getProviderPlugin(providerId);
    if (!plugin?.auth?.login) {
        throw new Error(`Provider ${providerId} does not support login`);
    }
    // Auth-only flow: do not mutate provider config here.
    setVerbose(Boolean(opts.verbose));
    const cfg = loadConfig();
    const accountId = opts.account?.trim() || resolveProviderDefaultAccountId({ plugin, cfg });
    await plugin.auth.login({
        cfg,
        accountId,
        runtime,
        verbose: Boolean(opts.verbose),
        providerInput,
    });
}
export async function runProviderLogout(opts, runtime = defaultRuntime) {
    const providerInput = opts.provider ?? DEFAULT_CHAT_PROVIDER;
    const providerId = normalizeProviderId(providerInput);
    if (!providerId) {
        throw new Error(`Unsupported provider: ${providerInput}`);
    }
    const plugin = getProviderPlugin(providerId);
    if (!plugin?.gateway?.logoutAccount) {
        throw new Error(`Provider ${providerId} does not support logout`);
    }
    // Auth-only flow: resolve account + clear session state only.
    const cfg = loadConfig();
    const accountId = opts.account?.trim() || resolveProviderDefaultAccountId({ plugin, cfg });
    const account = plugin.config.resolveAccount(cfg, accountId);
    await plugin.gateway.logoutAccount({
        cfg,
        accountId,
        account,
        runtime,
    });
}
