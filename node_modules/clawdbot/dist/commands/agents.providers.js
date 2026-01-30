import { resolveChannelDefaultAccountId } from "../channels/plugins/helpers.js";
import { getChannelPlugin, listChannelPlugins, normalizeChannelId, } from "../channels/plugins/index.js";
import { DEFAULT_ACCOUNT_ID } from "../routing/session-key.js";
function providerAccountKey(provider, accountId) {
    return `${provider}:${accountId ?? DEFAULT_ACCOUNT_ID}`;
}
function formatChannelAccountLabel(params) {
    const label = getChannelPlugin(params.provider)?.meta.label ?? params.provider;
    const account = params.name?.trim()
        ? `${params.accountId} (${params.name.trim()})`
        : params.accountId;
    return `${label} ${account}`;
}
function formatProviderState(entry) {
    const parts = [entry.state];
    if (entry.enabled === false && entry.state !== "disabled") {
        parts.push("disabled");
    }
    return parts.join(", ");
}
export async function buildProviderStatusIndex(cfg) {
    const map = new Map();
    for (const plugin of listChannelPlugins()) {
        const accountIds = plugin.config.listAccountIds(cfg);
        for (const accountId of accountIds) {
            const account = plugin.config.resolveAccount(cfg, accountId);
            const snapshot = plugin.config.describeAccount?.(account, cfg);
            const enabled = plugin.config.isEnabled
                ? plugin.config.isEnabled(account, cfg)
                : typeof snapshot?.enabled === "boolean"
                    ? snapshot.enabled
                    : account.enabled;
            const configured = plugin.config.isConfigured
                ? await plugin.config.isConfigured(account, cfg)
                : snapshot?.configured;
            const resolvedEnabled = typeof enabled === "boolean" ? enabled : true;
            const resolvedConfigured = typeof configured === "boolean" ? configured : true;
            const state = plugin.status?.resolveAccountState?.({
                account,
                cfg,
                configured: resolvedConfigured,
                enabled: resolvedEnabled,
            }) ??
                (typeof snapshot?.linked === "boolean"
                    ? snapshot.linked
                        ? "linked"
                        : "not linked"
                    : resolvedConfigured
                        ? "configured"
                        : "not configured");
            const name = snapshot?.name ?? account.name;
            map.set(providerAccountKey(plugin.id, accountId), {
                provider: plugin.id,
                accountId,
                name,
                state,
                enabled,
                configured,
            });
        }
    }
    return map;
}
function resolveDefaultAccountId(cfg, provider) {
    const plugin = getChannelPlugin(provider);
    if (!plugin)
        return DEFAULT_ACCOUNT_ID;
    return resolveChannelDefaultAccountId({ plugin, cfg });
}
function shouldShowProviderEntry(entry, cfg) {
    const plugin = getChannelPlugin(entry.provider);
    if (!plugin)
        return Boolean(entry.configured);
    if (plugin.meta.showConfigured === false) {
        const providerConfig = cfg[plugin.id];
        return Boolean(entry.configured) || Boolean(providerConfig);
    }
    return Boolean(entry.configured);
}
function formatProviderEntry(entry) {
    const label = formatChannelAccountLabel({
        provider: entry.provider,
        accountId: entry.accountId,
        name: entry.name,
    });
    return `${label}: ${formatProviderState(entry)}`;
}
export function summarizeBindings(cfg, bindings) {
    if (bindings.length === 0)
        return [];
    const seen = new Map();
    for (const binding of bindings) {
        const channel = normalizeChannelId(binding.match.channel);
        if (!channel)
            continue;
        const accountId = binding.match.accountId ?? resolveDefaultAccountId(cfg, channel);
        const key = providerAccountKey(channel, accountId);
        if (!seen.has(key)) {
            const label = formatChannelAccountLabel({
                provider: channel,
                accountId,
            });
            seen.set(key, label);
        }
    }
    return [...seen.values()];
}
export function listProvidersForAgent(params) {
    const allProviderEntries = [...params.providerStatus.values()];
    const providerLines = [];
    if (params.bindings.length > 0) {
        const seen = new Set();
        for (const binding of params.bindings) {
            const channel = normalizeChannelId(binding.match.channel);
            if (!channel)
                continue;
            const accountId = binding.match.accountId ?? resolveDefaultAccountId(params.cfg, channel);
            const key = providerAccountKey(channel, accountId);
            if (seen.has(key))
                continue;
            seen.add(key);
            const status = params.providerStatus.get(key);
            if (status) {
                providerLines.push(formatProviderEntry(status));
            }
            else {
                providerLines.push(`${formatChannelAccountLabel({ provider: channel, accountId })}: unknown`);
            }
        }
        return providerLines;
    }
    if (params.summaryIsDefault) {
        for (const entry of allProviderEntries) {
            if (shouldShowProviderEntry(entry, params.cfg)) {
                providerLines.push(formatProviderEntry(entry));
            }
        }
    }
    return providerLines;
}
