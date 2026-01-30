import { normalizeGroupActivation } from "../../../auto-reply/group-activation.js";
import { resolveChannelGroupPolicy, resolveChannelGroupRequireMention, } from "../../../config/group-policy.js";
import { loadSessionStore, resolveGroupSessionKey, resolveStorePath, } from "../../../config/sessions.js";
export function resolveGroupPolicyFor(cfg, conversationId) {
    const groupId = resolveGroupSessionKey({
        From: conversationId,
        ChatType: "group",
        Provider: "whatsapp",
    })?.id;
    return resolveChannelGroupPolicy({
        cfg,
        channel: "whatsapp",
        groupId: groupId ?? conversationId,
    });
}
export function resolveGroupRequireMentionFor(cfg, conversationId) {
    const groupId = resolveGroupSessionKey({
        From: conversationId,
        ChatType: "group",
        Provider: "whatsapp",
    })?.id;
    return resolveChannelGroupRequireMention({
        cfg,
        channel: "whatsapp",
        groupId: groupId ?? conversationId,
    });
}
export function resolveGroupActivationFor(params) {
    const storePath = resolveStorePath(params.cfg.session?.store, {
        agentId: params.agentId,
    });
    const store = loadSessionStore(storePath);
    const entry = store[params.sessionKey];
    const requireMention = resolveGroupRequireMentionFor(params.cfg, params.conversationId);
    const defaultActivation = requireMention === false ? "always" : "mention";
    return normalizeGroupActivation(entry?.groupActivation) ?? defaultActivation;
}
