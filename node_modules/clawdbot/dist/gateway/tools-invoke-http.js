import { createClawdbotTools } from "../agents/clawdbot-tools.js";
import { filterToolsByPolicy, resolveEffectiveToolPolicy, resolveGroupToolPolicy, resolveSubagentToolPolicy, } from "../agents/pi-tools.policy.js";
import { buildPluginToolGroups, collectExplicitAllowlist, expandPolicyWithPluginGroups, normalizeToolName, resolveToolProfilePolicy, stripPluginOnlyAllowlist, } from "../agents/tool-policy.js";
import { loadConfig } from "../config/config.js";
import { resolveMainSessionKey } from "../config/sessions.js";
import { logWarn } from "../logger.js";
import { getPluginToolMeta } from "../plugins/tools.js";
import { isSubagentSessionKey } from "../routing/session-key.js";
import { normalizeMessageChannel } from "../utils/message-channel.js";
import { authorizeGatewayConnect } from "./auth.js";
import { getBearerToken, getHeader } from "./http-utils.js";
import { readJsonBodyOrError, sendInvalidRequest, sendJson, sendMethodNotAllowed, sendUnauthorized, } from "./http-common.js";
const DEFAULT_BODY_BYTES = 2 * 1024 * 1024;
function resolveSessionKeyFromBody(body) {
    if (typeof body.sessionKey === "string" && body.sessionKey.trim())
        return body.sessionKey.trim();
    return undefined;
}
function mergeActionIntoArgsIfSupported(params) {
    const { toolSchema, action, args } = params;
    if (!action)
        return args;
    if (args.action !== undefined)
        return args;
    // TypeBox schemas are plain objects; many tools define an `action` property.
    const schemaObj = toolSchema;
    const hasAction = Boolean(schemaObj &&
        typeof schemaObj === "object" &&
        schemaObj.properties &&
        "action" in schemaObj.properties);
    if (!hasAction)
        return args;
    return { ...args, action };
}
export async function handleToolsInvokeHttpRequest(req, res, opts) {
    const url = new URL(req.url ?? "/", `http://${req.headers.host ?? "localhost"}`);
    if (url.pathname !== "/tools/invoke")
        return false;
    if (req.method !== "POST") {
        sendMethodNotAllowed(res, "POST");
        return true;
    }
    const cfg = loadConfig();
    const token = getBearerToken(req);
    const authResult = await authorizeGatewayConnect({
        auth: opts.auth,
        connectAuth: token ? { token, password: token } : null,
        req,
        trustedProxies: opts.trustedProxies ?? cfg.gateway?.trustedProxies,
    });
    if (!authResult.ok) {
        sendUnauthorized(res);
        return true;
    }
    const bodyUnknown = await readJsonBodyOrError(req, res, opts.maxBodyBytes ?? DEFAULT_BODY_BYTES);
    if (bodyUnknown === undefined)
        return true;
    const body = (bodyUnknown ?? {});
    const toolName = typeof body.tool === "string" ? body.tool.trim() : "";
    if (!toolName) {
        sendInvalidRequest(res, "tools.invoke requires body.tool");
        return true;
    }
    const action = typeof body.action === "string" ? body.action.trim() : undefined;
    const argsRaw = body.args;
    const args = (argsRaw && typeof argsRaw === "object" && !Array.isArray(argsRaw)
        ? argsRaw
        : {});
    const rawSessionKey = resolveSessionKeyFromBody(body);
    const sessionKey = !rawSessionKey || rawSessionKey === "main" ? resolveMainSessionKey(cfg) : rawSessionKey;
    // Resolve message channel/account hints (optional headers) for policy inheritance.
    const messageChannel = normalizeMessageChannel(getHeader(req, "x-clawdbot-message-channel") ?? "");
    const accountId = getHeader(req, "x-clawdbot-account-id")?.trim() || undefined;
    const { agentId, globalPolicy, globalProviderPolicy, agentPolicy, agentProviderPolicy, profile, providerProfile, } = resolveEffectiveToolPolicy({ config: cfg, sessionKey });
    const profilePolicy = resolveToolProfilePolicy(profile);
    const providerProfilePolicy = resolveToolProfilePolicy(providerProfile);
    const groupPolicy = resolveGroupToolPolicy({
        config: cfg,
        sessionKey,
        messageProvider: messageChannel ?? undefined,
        accountId: accountId ?? null,
    });
    const subagentPolicy = isSubagentSessionKey(sessionKey)
        ? resolveSubagentToolPolicy(cfg)
        : undefined;
    // Build tool list (core + plugin tools).
    const allTools = createClawdbotTools({
        agentSessionKey: sessionKey,
        agentChannel: messageChannel ?? undefined,
        agentAccountId: accountId,
        config: cfg,
        pluginToolAllowlist: collectExplicitAllowlist([
            profilePolicy,
            providerProfilePolicy,
            globalPolicy,
            globalProviderPolicy,
            agentPolicy,
            agentProviderPolicy,
            groupPolicy,
            subagentPolicy,
        ]),
    });
    const coreToolNames = new Set(allTools
        .filter((tool) => !getPluginToolMeta(tool))
        .map((tool) => normalizeToolName(tool.name))
        .filter(Boolean));
    const pluginGroups = buildPluginToolGroups({
        tools: allTools,
        toolMeta: (tool) => getPluginToolMeta(tool),
    });
    const resolvePolicy = (policy, label) => {
        const resolved = stripPluginOnlyAllowlist(policy, pluginGroups, coreToolNames);
        if (resolved.unknownAllowlist.length > 0) {
            const entries = resolved.unknownAllowlist.join(", ");
            const suffix = resolved.strippedAllowlist
                ? "Ignoring allowlist so core tools remain available."
                : "These entries won't match any tool unless the plugin is enabled.";
            logWarn(`tools: ${label} allowlist contains unknown entries (${entries}). ${suffix}`);
        }
        return expandPolicyWithPluginGroups(resolved.policy, pluginGroups);
    };
    const profilePolicyExpanded = resolvePolicy(profilePolicy, profile ? `tools.profile (${profile})` : "tools.profile");
    const providerProfileExpanded = resolvePolicy(providerProfilePolicy, providerProfile ? `tools.byProvider.profile (${providerProfile})` : "tools.byProvider.profile");
    const globalPolicyExpanded = resolvePolicy(globalPolicy, "tools.allow");
    const globalProviderExpanded = resolvePolicy(globalProviderPolicy, "tools.byProvider.allow");
    const agentPolicyExpanded = resolvePolicy(agentPolicy, agentId ? `agents.${agentId}.tools.allow` : "agent tools.allow");
    const agentProviderExpanded = resolvePolicy(agentProviderPolicy, agentId ? `agents.${agentId}.tools.byProvider.allow` : "agent tools.byProvider.allow");
    const groupPolicyExpanded = resolvePolicy(groupPolicy, "group tools.allow");
    const subagentPolicyExpanded = expandPolicyWithPluginGroups(subagentPolicy, pluginGroups);
    const toolsFiltered = profilePolicyExpanded
        ? filterToolsByPolicy(allTools, profilePolicyExpanded)
        : allTools;
    const providerProfileFiltered = providerProfileExpanded
        ? filterToolsByPolicy(toolsFiltered, providerProfileExpanded)
        : toolsFiltered;
    const globalFiltered = globalPolicyExpanded
        ? filterToolsByPolicy(providerProfileFiltered, globalPolicyExpanded)
        : providerProfileFiltered;
    const globalProviderFiltered = globalProviderExpanded
        ? filterToolsByPolicy(globalFiltered, globalProviderExpanded)
        : globalFiltered;
    const agentFiltered = agentPolicyExpanded
        ? filterToolsByPolicy(globalProviderFiltered, agentPolicyExpanded)
        : globalProviderFiltered;
    const agentProviderFiltered = agentProviderExpanded
        ? filterToolsByPolicy(agentFiltered, agentProviderExpanded)
        : agentFiltered;
    const groupFiltered = groupPolicyExpanded
        ? filterToolsByPolicy(agentProviderFiltered, groupPolicyExpanded)
        : agentProviderFiltered;
    const subagentFiltered = subagentPolicyExpanded
        ? filterToolsByPolicy(groupFiltered, subagentPolicyExpanded)
        : groupFiltered;
    const tool = subagentFiltered.find((t) => t.name === toolName);
    if (!tool) {
        sendJson(res, 404, {
            ok: false,
            error: { type: "not_found", message: `Tool not available: ${toolName}` },
        });
        return true;
    }
    try {
        const toolArgs = mergeActionIntoArgsIfSupported({
            toolSchema: tool.parameters,
            action,
            args,
        });
        const result = await tool.execute?.(`http-${Date.now()}`, toolArgs);
        sendJson(res, 200, { ok: true, result });
    }
    catch (err) {
        sendJson(res, 400, {
            ok: false,
            error: { type: "tool_error", message: err instanceof Error ? err.message : String(err) },
        });
    }
    return true;
}
