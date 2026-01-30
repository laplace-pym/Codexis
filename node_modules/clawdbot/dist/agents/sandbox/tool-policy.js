import { resolveAgentConfig } from "../agent-scope.js";
import { expandToolGroups } from "../tool-policy.js";
import { DEFAULT_TOOL_ALLOW, DEFAULT_TOOL_DENY } from "./constants.js";
function compilePattern(pattern) {
    const normalized = pattern.trim().toLowerCase();
    if (!normalized)
        return { kind: "exact", value: "" };
    if (normalized === "*")
        return { kind: "all" };
    if (!normalized.includes("*"))
        return { kind: "exact", value: normalized };
    const escaped = normalized.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    return {
        kind: "regex",
        value: new RegExp(`^${escaped.replaceAll("\\*", ".*")}$`),
    };
}
function compilePatterns(patterns) {
    if (!Array.isArray(patterns))
        return [];
    return expandToolGroups(patterns)
        .map(compilePattern)
        .filter((pattern) => pattern.kind !== "exact" || pattern.value);
}
function matchesAny(name, patterns) {
    for (const pattern of patterns) {
        if (pattern.kind === "all")
            return true;
        if (pattern.kind === "exact" && name === pattern.value)
            return true;
        if (pattern.kind === "regex" && pattern.value.test(name))
            return true;
    }
    return false;
}
export function isToolAllowed(policy, name) {
    const normalized = name.trim().toLowerCase();
    const deny = compilePatterns(policy.deny);
    if (matchesAny(normalized, deny))
        return false;
    const allow = compilePatterns(policy.allow);
    if (allow.length === 0)
        return true;
    return matchesAny(normalized, allow);
}
export function resolveSandboxToolPolicyForAgent(cfg, agentId) {
    const agentConfig = cfg && agentId ? resolveAgentConfig(cfg, agentId) : undefined;
    const agentAllow = agentConfig?.tools?.sandbox?.tools?.allow;
    const agentDeny = agentConfig?.tools?.sandbox?.tools?.deny;
    const globalAllow = cfg?.tools?.sandbox?.tools?.allow;
    const globalDeny = cfg?.tools?.sandbox?.tools?.deny;
    const allowSource = Array.isArray(agentAllow)
        ? {
            source: "agent",
            key: "agents.list[].tools.sandbox.tools.allow",
        }
        : Array.isArray(globalAllow)
            ? {
                source: "global",
                key: "tools.sandbox.tools.allow",
            }
            : {
                source: "default",
                key: "tools.sandbox.tools.allow",
            };
    const denySource = Array.isArray(agentDeny)
        ? {
            source: "agent",
            key: "agents.list[].tools.sandbox.tools.deny",
        }
        : Array.isArray(globalDeny)
            ? {
                source: "global",
                key: "tools.sandbox.tools.deny",
            }
            : {
                source: "default",
                key: "tools.sandbox.tools.deny",
            };
    const deny = Array.isArray(agentDeny)
        ? agentDeny
        : Array.isArray(globalDeny)
            ? globalDeny
            : [...DEFAULT_TOOL_DENY];
    const allow = Array.isArray(agentAllow)
        ? agentAllow
        : Array.isArray(globalAllow)
            ? globalAllow
            : [...DEFAULT_TOOL_ALLOW];
    const expandedDeny = expandToolGroups(deny);
    let expandedAllow = expandToolGroups(allow);
    // `image` is essential for multimodal workflows; always include it in sandboxed
    // sessions unless explicitly denied.
    if (!expandedDeny.map((v) => v.toLowerCase()).includes("image") &&
        !expandedAllow.map((v) => v.toLowerCase()).includes("image")) {
        expandedAllow = [...expandedAllow, "image"];
    }
    return {
        allow: expandedAllow,
        deny: expandedDeny,
        sources: {
            allow: allowSource,
            deny: denySource,
        },
    };
}
