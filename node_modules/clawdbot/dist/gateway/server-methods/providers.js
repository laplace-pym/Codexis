import { loadConfig, readConfigFileSnapshot } from "../../config/config.js";
import { getProviderActivity } from "../../infra/provider-activity.js";
import { resolveProviderDefaultAccountId } from "../../providers/plugins/helpers.js";
import { getProviderPlugin, listProviderPlugins, normalizeProviderId, } from "../../providers/plugins/index.js";
import { buildProviderAccountSnapshot } from "../../providers/plugins/status.js";
import { DEFAULT_ACCOUNT_ID } from "../../routing/session-key.js";
import { defaultRuntime } from "../../runtime.js";
import { ErrorCodes, errorShape, formatValidationErrors, validateProvidersLogoutParams, validateProvidersStatusParams, } from "../protocol/index.js";
import { formatForLog } from "../ws-log.js";
export async function logoutProviderAccount(params) {
    const resolvedAccountId = params.accountId?.trim() ||
        params.plugin.config.defaultAccountId?.(params.cfg) ||
        params.plugin.config.listAccountIds(params.cfg)[0] ||
        DEFAULT_ACCOUNT_ID;
    const account = params.plugin.config.resolveAccount(params.cfg, resolvedAccountId);
    await params.context.stopProvider(params.providerId, resolvedAccountId);
    const result = await params.plugin.gateway?.logoutAccount?.({
        cfg: params.cfg,
        accountId: resolvedAccountId,
        account,
        runtime: defaultRuntime,
    });
    if (!result) {
        throw new Error(`Provider ${params.providerId} does not support logout`);
    }
    const cleared = Boolean(result.cleared);
    const loggedOut = typeof result.loggedOut === "boolean" ? result.loggedOut : cleared;
    if (loggedOut) {
        params.context.markProviderLoggedOut(params.providerId, true, resolvedAccountId);
    }
    return {
        provider: params.providerId,
        accountId: resolvedAccountId,
        ...result,
        cleared,
    };
}
export const providersHandlers = {
    "providers.status": async ({ params, respond, context }) => {
        if (!validateProvidersStatusParams(params)) {
            respond(false, undefined, errorShape(ErrorCodes.INVALID_REQUEST, `invalid providers.status params: ${formatValidationErrors(validateProvidersStatusParams.errors)}`));
            return;
        }
        const probe = params.probe === true;
        const timeoutMsRaw = params.timeoutMs;
        const timeoutMs = typeof timeoutMsRaw === "number" ? Math.max(1000, timeoutMsRaw) : 10_000;
        const cfg = loadConfig();
        const runtime = context.getRuntimeSnapshot();
        const plugins = listProviderPlugins();
        const pluginMap = new Map(plugins.map((plugin) => [plugin.id, plugin]));
        const resolveRuntimeSnapshot = (providerId, accountId, defaultAccountId) => {
            const accounts = runtime.providerAccounts[providerId];
            const defaultRuntime = runtime.providers[providerId];
            const raw = accounts?.[accountId] ??
                (accountId === defaultAccountId ? defaultRuntime : undefined);
            if (!raw)
                return undefined;
            return raw;
        };
        const isAccountEnabled = (plugin, account) => plugin.config.isEnabled
            ? plugin.config.isEnabled(account, cfg)
            : !account ||
                typeof account !== "object" ||
                account.enabled !== false;
        const buildProviderAccounts = async (providerId) => {
            const plugin = pluginMap.get(providerId);
            if (!plugin) {
                return {
                    accounts: [],
                    defaultAccountId: DEFAULT_ACCOUNT_ID,
                    defaultAccount: undefined,
                    resolvedAccounts: {},
                };
            }
            const accountIds = plugin.config.listAccountIds(cfg);
            const defaultAccountId = resolveProviderDefaultAccountId({
                plugin,
                cfg,
                accountIds,
            });
            const accounts = [];
            const resolvedAccounts = {};
            for (const accountId of accountIds) {
                const account = plugin.config.resolveAccount(cfg, accountId);
                const enabled = isAccountEnabled(plugin, account);
                resolvedAccounts[accountId] = account;
                let probeResult;
                let lastProbeAt = null;
                if (probe && enabled && plugin.status?.probeAccount) {
                    let configured = true;
                    if (plugin.config.isConfigured) {
                        configured = await plugin.config.isConfigured(account, cfg);
                    }
                    if (configured) {
                        probeResult = await plugin.status.probeAccount({
                            account,
                            timeoutMs,
                            cfg,
                        });
                        lastProbeAt = Date.now();
                    }
                }
                let auditResult;
                if (probe && enabled && plugin.status?.auditAccount) {
                    let configured = true;
                    if (plugin.config.isConfigured) {
                        configured = await plugin.config.isConfigured(account, cfg);
                    }
                    if (configured) {
                        auditResult = await plugin.status.auditAccount({
                            account,
                            timeoutMs,
                            cfg,
                            probe: probeResult,
                        });
                    }
                }
                const runtimeSnapshot = resolveRuntimeSnapshot(providerId, accountId, defaultAccountId);
                const snapshot = await buildProviderAccountSnapshot({
                    plugin,
                    cfg,
                    accountId,
                    runtime: runtimeSnapshot,
                    probe: probeResult,
                    audit: auditResult,
                });
                if (lastProbeAt)
                    snapshot.lastProbeAt = lastProbeAt;
                const activity = getProviderActivity({
                    provider: providerId,
                    accountId,
                });
                if (snapshot.lastInboundAt == null) {
                    snapshot.lastInboundAt = activity.inboundAt;
                }
                if (snapshot.lastOutboundAt == null) {
                    snapshot.lastOutboundAt = activity.outboundAt;
                }
                accounts.push(snapshot);
            }
            const defaultAccount = accounts.find((entry) => entry.accountId === defaultAccountId) ??
                accounts[0];
            return { accounts, defaultAccountId, defaultAccount, resolvedAccounts };
        };
        const payload = {
            ts: Date.now(),
            providerOrder: plugins.map((plugin) => plugin.id),
            providerLabels: Object.fromEntries(plugins.map((plugin) => [plugin.id, plugin.meta.label])),
            providers: {},
            providerAccounts: {},
            providerDefaultAccountId: {},
        };
        const providersMap = payload.providers;
        const accountsMap = payload.providerAccounts;
        const defaultAccountIdMap = payload.providerDefaultAccountId;
        for (const plugin of plugins) {
            const { accounts, defaultAccountId, defaultAccount, resolvedAccounts } = await buildProviderAccounts(plugin.id);
            const fallbackAccount = resolvedAccounts[defaultAccountId] ??
                plugin.config.resolveAccount(cfg, defaultAccountId);
            const summary = plugin.status?.buildProviderSummary
                ? await plugin.status.buildProviderSummary({
                    account: fallbackAccount,
                    cfg,
                    defaultAccountId,
                    snapshot: defaultAccount ??
                        {
                            accountId: defaultAccountId,
                        },
                })
                : {
                    configured: defaultAccount?.configured ?? false,
                };
            providersMap[plugin.id] = summary;
            accountsMap[plugin.id] = accounts;
            defaultAccountIdMap[plugin.id] = defaultAccountId;
        }
        respond(true, payload, undefined);
    },
    "providers.logout": async ({ params, respond, context }) => {
        if (!validateProvidersLogoutParams(params)) {
            respond(false, undefined, errorShape(ErrorCodes.INVALID_REQUEST, `invalid providers.logout params: ${formatValidationErrors(validateProvidersLogoutParams.errors)}`));
            return;
        }
        const rawProvider = params.provider;
        const providerId = typeof rawProvider === "string" ? normalizeProviderId(rawProvider) : null;
        if (!providerId) {
            respond(false, undefined, errorShape(ErrorCodes.INVALID_REQUEST, "invalid providers.logout provider"));
            return;
        }
        const plugin = getProviderPlugin(providerId);
        if (!plugin?.gateway?.logoutAccount) {
            respond(false, undefined, errorShape(ErrorCodes.INVALID_REQUEST, `provider ${providerId} does not support logout`));
            return;
        }
        const accountIdRaw = params.accountId;
        const accountId = typeof accountIdRaw === "string" ? accountIdRaw.trim() : undefined;
        const snapshot = await readConfigFileSnapshot();
        if (!snapshot.valid) {
            respond(false, undefined, errorShape(ErrorCodes.INVALID_REQUEST, "config invalid; fix it before logging out"));
            return;
        }
        try {
            const payload = await logoutProviderAccount({
                providerId,
                accountId,
                cfg: snapshot.config ?? {},
                context,
                plugin,
            });
            respond(true, payload, undefined);
        }
        catch (err) {
            respond(false, undefined, errorShape(ErrorCodes.UNAVAILABLE, formatForLog(err)));
        }
    },
};
