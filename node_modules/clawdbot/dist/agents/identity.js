import { resolveAgentConfig } from "./agent-scope.js";
const DEFAULT_ACK_REACTION = "👀";
export function resolveAgentIdentity(cfg, agentId) {
    return resolveAgentConfig(cfg, agentId)?.identity;
}
export function resolveAckReaction(cfg, agentId) {
    const configured = cfg.messages?.ackReaction;
    if (configured !== undefined)
        return configured.trim();
    const emoji = resolveAgentIdentity(cfg, agentId)?.emoji?.trim();
    return emoji || DEFAULT_ACK_REACTION;
}
export function resolveIdentityNamePrefix(cfg, agentId) {
    const name = resolveAgentIdentity(cfg, agentId)?.name?.trim();
    if (!name)
        return undefined;
    return `[${name}]`;
}
/** Returns just the identity name (without brackets) for template context. */
export function resolveIdentityName(cfg, agentId) {
    return resolveAgentIdentity(cfg, agentId)?.name?.trim() || undefined;
}
export function resolveMessagePrefix(cfg, agentId, opts) {
    const configured = opts?.configured ?? cfg.messages?.messagePrefix;
    if (configured !== undefined)
        return configured;
    const hasAllowFrom = opts?.hasAllowFrom === true;
    if (hasAllowFrom)
        return "";
    return resolveIdentityNamePrefix(cfg, agentId) ?? opts?.fallback ?? "[clawdbot]";
}
export function resolveResponsePrefix(cfg, agentId) {
    const configured = cfg.messages?.responsePrefix;
    if (configured !== undefined) {
        if (configured === "auto") {
            return resolveIdentityNamePrefix(cfg, agentId);
        }
        return configured;
    }
    return undefined;
}
export function resolveEffectiveMessagesConfig(cfg, agentId, opts) {
    return {
        messagePrefix: resolveMessagePrefix(cfg, agentId, {
            hasAllowFrom: opts?.hasAllowFrom,
            fallback: opts?.fallbackMessagePrefix,
        }),
        responsePrefix: resolveResponsePrefix(cfg, agentId),
    };
}
export function resolveHumanDelayConfig(cfg, agentId) {
    const defaults = cfg.agents?.defaults?.humanDelay;
    const overrides = resolveAgentConfig(cfg, agentId)?.humanDelay;
    if (!defaults && !overrides)
        return undefined;
    return {
        mode: overrides?.mode ?? defaults?.mode,
        minMs: overrides?.minMs ?? defaults?.minMs,
        maxMs: overrides?.maxMs ?? defaults?.maxMs,
    };
}
