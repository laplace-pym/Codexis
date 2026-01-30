import { resolveDefaultAgentId } from "../agents/agent-scope.js";
import { normalizeChatChannelId } from "../channels/registry.js";
import { normalizeAccountId, normalizeAgentId } from "./session-key.js";
function normalizeBindingChannelId(raw) {
    const normalized = normalizeChatChannelId(raw);
    if (normalized)
        return normalized;
    const fallback = (raw ?? "").trim().toLowerCase();
    return fallback || null;
}
export function listBindings(cfg) {
    return Array.isArray(cfg.bindings) ? cfg.bindings : [];
}
export function listBoundAccountIds(cfg, channelId) {
    const normalizedChannel = normalizeBindingChannelId(channelId);
    if (!normalizedChannel)
        return [];
    const ids = new Set();
    for (const binding of listBindings(cfg)) {
        if (!binding || typeof binding !== "object")
            continue;
        const match = binding.match;
        if (!match || typeof match !== "object")
            continue;
        const channel = normalizeBindingChannelId(match.channel);
        if (!channel || channel !== normalizedChannel)
            continue;
        const accountId = typeof match.accountId === "string" ? match.accountId.trim() : "";
        if (!accountId || accountId === "*")
            continue;
        ids.add(normalizeAccountId(accountId));
    }
    return Array.from(ids).sort((a, b) => a.localeCompare(b));
}
export function resolveDefaultAgentBoundAccountId(cfg, channelId) {
    const normalizedChannel = normalizeBindingChannelId(channelId);
    if (!normalizedChannel)
        return null;
    const defaultAgentId = normalizeAgentId(resolveDefaultAgentId(cfg));
    for (const binding of listBindings(cfg)) {
        if (!binding || typeof binding !== "object")
            continue;
        if (normalizeAgentId(binding.agentId) !== defaultAgentId)
            continue;
        const match = binding.match;
        if (!match || typeof match !== "object")
            continue;
        const channel = normalizeBindingChannelId(match.channel);
        if (!channel || channel !== normalizedChannel)
            continue;
        const accountId = typeof match.accountId === "string" ? match.accountId.trim() : "";
        if (!accountId || accountId === "*")
            continue;
        return normalizeAccountId(accountId);
    }
    return null;
}
export function buildChannelAccountBindings(cfg) {
    const map = new Map();
    for (const binding of listBindings(cfg)) {
        if (!binding || typeof binding !== "object")
            continue;
        const match = binding.match;
        if (!match || typeof match !== "object")
            continue;
        const channelId = normalizeBindingChannelId(match.channel);
        if (!channelId)
            continue;
        const accountId = typeof match.accountId === "string" ? match.accountId.trim() : "";
        if (!accountId || accountId === "*")
            continue;
        const agentId = normalizeAgentId(binding.agentId);
        const byAgent = map.get(channelId) ?? new Map();
        const list = byAgent.get(agentId) ?? [];
        const normalizedAccountId = normalizeAccountId(accountId);
        if (!list.includes(normalizedAccountId))
            list.push(normalizedAccountId);
        byAgent.set(agentId, list);
        map.set(channelId, byAgent);
    }
    return map;
}
export function resolvePreferredAccountId(params) {
    if (params.boundAccounts.length > 0)
        return params.boundAccounts[0];
    return params.defaultAccountId;
}
