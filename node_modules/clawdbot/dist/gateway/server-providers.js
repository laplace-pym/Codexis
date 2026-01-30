import { formatErrorMessage } from "../infra/errors.js";
import { resolveProviderDefaultAccountId } from "../providers/plugins/helpers.js";
import { getProviderPlugin, listProviderPlugins, } from "../providers/plugins/index.js";
import { DEFAULT_ACCOUNT_ID } from "../routing/session-key.js";
function createRuntimeStore() {
    return {
        aborts: new Map(),
        tasks: new Map(),
        runtimes: new Map(),
    };
}
function isAccountEnabled(account) {
    if (!account || typeof account !== "object")
        return true;
    const enabled = account.enabled;
    return enabled !== false;
}
function resolveDefaultRuntime(providerId) {
    const plugin = getProviderPlugin(providerId);
    return plugin?.status?.defaultRuntime ?? { accountId: DEFAULT_ACCOUNT_ID };
}
function cloneDefaultRuntime(providerId, accountId) {
    return { ...resolveDefaultRuntime(providerId), accountId };
}
// Provider docking: lifecycle hooks (`plugin.gateway`) flow through this manager.
export function createProviderManager(opts) {
    const { loadConfig, providerLogs, providerRuntimeEnvs } = opts;
    const providerStores = new Map();
    const getStore = (providerId) => {
        const existing = providerStores.get(providerId);
        if (existing)
            return existing;
        const next = createRuntimeStore();
        providerStores.set(providerId, next);
        return next;
    };
    const getRuntime = (providerId, accountId) => {
        const store = getStore(providerId);
        return (store.runtimes.get(accountId) ??
            cloneDefaultRuntime(providerId, accountId));
    };
    const setRuntime = (providerId, accountId, patch) => {
        const store = getStore(providerId);
        const current = getRuntime(providerId, accountId);
        const next = { ...current, ...patch, accountId };
        store.runtimes.set(accountId, next);
        return next;
    };
    const startProvider = async (providerId, accountId) => {
        const plugin = getProviderPlugin(providerId);
        const startAccount = plugin?.gateway?.startAccount;
        if (!startAccount)
            return;
        const cfg = loadConfig();
        const store = getStore(providerId);
        const accountIds = accountId
            ? [accountId]
            : plugin.config.listAccountIds(cfg);
        if (accountIds.length === 0)
            return;
        await Promise.all(accountIds.map(async (id) => {
            if (store.tasks.has(id))
                return;
            const account = plugin.config.resolveAccount(cfg, id);
            const enabled = plugin.config.isEnabled
                ? plugin.config.isEnabled(account, cfg)
                : isAccountEnabled(account);
            if (!enabled) {
                setRuntime(providerId, id, {
                    accountId: id,
                    running: false,
                    lastError: plugin.config.disabledReason?.(account, cfg) ?? "disabled",
                });
                return;
            }
            let configured = true;
            if (plugin.config.isConfigured) {
                configured = await plugin.config.isConfigured(account, cfg);
            }
            if (!configured) {
                setRuntime(providerId, id, {
                    accountId: id,
                    running: false,
                    lastError: plugin.config.unconfiguredReason?.(account, cfg) ??
                        "not configured",
                });
                return;
            }
            const abort = new AbortController();
            store.aborts.set(id, abort);
            setRuntime(providerId, id, {
                accountId: id,
                running: true,
                lastStartAt: Date.now(),
                lastError: null,
            });
            const log = providerLogs[providerId];
            const task = startAccount({
                cfg,
                accountId: id,
                account,
                runtime: providerRuntimeEnvs[providerId],
                abortSignal: abort.signal,
                log,
                getStatus: () => getRuntime(providerId, id),
                setStatus: (next) => setRuntime(providerId, id, next),
            });
            const tracked = Promise.resolve(task)
                .catch((err) => {
                const message = formatErrorMessage(err);
                setRuntime(providerId, id, { accountId: id, lastError: message });
                log.error?.(`[${id}] provider exited: ${message}`);
            })
                .finally(() => {
                store.aborts.delete(id);
                store.tasks.delete(id);
                setRuntime(providerId, id, {
                    accountId: id,
                    running: false,
                    lastStopAt: Date.now(),
                });
            });
            store.tasks.set(id, tracked);
        }));
    };
    const stopProvider = async (providerId, accountId) => {
        const plugin = getProviderPlugin(providerId);
        const cfg = loadConfig();
        const store = getStore(providerId);
        const knownIds = new Set([
            ...store.aborts.keys(),
            ...store.tasks.keys(),
            ...(plugin ? plugin.config.listAccountIds(cfg) : []),
        ]);
        if (accountId) {
            knownIds.clear();
            knownIds.add(accountId);
        }
        await Promise.all(Array.from(knownIds.values()).map(async (id) => {
            const abort = store.aborts.get(id);
            const task = store.tasks.get(id);
            if (!abort && !task && !plugin?.gateway?.stopAccount)
                return;
            abort?.abort();
            if (plugin?.gateway?.stopAccount) {
                const account = plugin.config.resolveAccount(cfg, id);
                await plugin.gateway.stopAccount({
                    cfg,
                    accountId: id,
                    account,
                    runtime: providerRuntimeEnvs[providerId],
                    abortSignal: abort?.signal ?? new AbortController().signal,
                    log: providerLogs[providerId],
                    getStatus: () => getRuntime(providerId, id),
                    setStatus: (next) => setRuntime(providerId, id, next),
                });
            }
            try {
                await task;
            }
            catch {
                // ignore
            }
            store.aborts.delete(id);
            store.tasks.delete(id);
            setRuntime(providerId, id, {
                accountId: id,
                running: false,
                lastStopAt: Date.now(),
            });
        }));
    };
    const startProviders = async () => {
        for (const plugin of listProviderPlugins()) {
            await startProvider(plugin.id);
        }
    };
    const markProviderLoggedOut = (providerId, cleared, accountId) => {
        const plugin = getProviderPlugin(providerId);
        if (!plugin)
            return;
        const cfg = loadConfig();
        const resolvedId = accountId ??
            resolveProviderDefaultAccountId({
                plugin,
                cfg,
            });
        const current = getRuntime(providerId, resolvedId);
        const next = {
            accountId: resolvedId,
            running: false,
            lastError: cleared ? "logged out" : current.lastError,
        };
        if (typeof current.connected === "boolean") {
            next.connected = false;
        }
        setRuntime(providerId, resolvedId, next);
    };
    const getRuntimeSnapshot = () => {
        const cfg = loadConfig();
        const providers = {};
        const providerAccounts = {};
        for (const plugin of listProviderPlugins()) {
            const store = getStore(plugin.id);
            const accountIds = plugin.config.listAccountIds(cfg);
            const defaultAccountId = resolveProviderDefaultAccountId({
                plugin,
                cfg,
                accountIds,
            });
            const accounts = {};
            for (const id of accountIds) {
                const account = plugin.config.resolveAccount(cfg, id);
                const enabled = plugin.config.isEnabled
                    ? plugin.config.isEnabled(account, cfg)
                    : isAccountEnabled(account);
                const described = plugin.config.describeAccount?.(account, cfg);
                const configured = described?.configured;
                const current = store.runtimes.get(id) ?? cloneDefaultRuntime(plugin.id, id);
                const next = { ...current, accountId: id };
                if (!next.running) {
                    if (!enabled) {
                        next.lastError ??=
                            plugin.config.disabledReason?.(account, cfg) ?? "disabled";
                    }
                    else if (configured === false) {
                        next.lastError ??=
                            plugin.config.unconfiguredReason?.(account, cfg) ??
                                "not configured";
                    }
                }
                accounts[id] = next;
            }
            const defaultAccount = accounts[defaultAccountId] ??
                cloneDefaultRuntime(plugin.id, defaultAccountId);
            providers[plugin.id] = defaultAccount;
            providerAccounts[plugin.id] = accounts;
        }
        return { providers, providerAccounts };
    };
    return {
        getRuntimeSnapshot,
        startProviders,
        startProvider,
        stopProvider,
        markProviderLoggedOut,
    };
}
