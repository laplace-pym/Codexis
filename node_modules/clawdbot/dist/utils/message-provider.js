import { GATEWAY_CLIENT_MODES, GATEWAY_CLIENT_NAMES, normalizeGatewayClientMode, normalizeGatewayClientName, } from "../gateway/protocol/client-info.js";
import { listChatProviderAliases, normalizeChatProviderId, PROVIDER_IDS, } from "../providers/registry.js";
export const INTERNAL_MESSAGE_PROVIDER = "webchat";
export { GATEWAY_CLIENT_NAMES, GATEWAY_CLIENT_MODES };
export { normalizeGatewayClientName, normalizeGatewayClientMode };
export function isGatewayCliClient(client) {
    return normalizeGatewayClientMode(client?.mode) === GATEWAY_CLIENT_MODES.CLI;
}
export function isInternalMessageProvider(raw) {
    return normalizeMessageProvider(raw) === INTERNAL_MESSAGE_PROVIDER;
}
export function isWebchatClient(client) {
    const mode = normalizeGatewayClientMode(client?.mode);
    if (mode === GATEWAY_CLIENT_MODES.WEBCHAT)
        return true;
    return (normalizeGatewayClientName(client?.id) === GATEWAY_CLIENT_NAMES.WEBCHAT_UI);
}
export function normalizeMessageProvider(raw) {
    const normalized = raw?.trim().toLowerCase();
    if (!normalized)
        return undefined;
    if (normalized === INTERNAL_MESSAGE_PROVIDER)
        return INTERNAL_MESSAGE_PROVIDER;
    return normalizeChatProviderId(normalized) ?? normalized;
}
export const DELIVERABLE_MESSAGE_PROVIDERS = PROVIDER_IDS;
export const GATEWAY_MESSAGE_PROVIDERS = [
    ...DELIVERABLE_MESSAGE_PROVIDERS,
    INTERNAL_MESSAGE_PROVIDER,
];
export const GATEWAY_AGENT_PROVIDER_ALIASES = listChatProviderAliases();
export const GATEWAY_AGENT_PROVIDER_VALUES = Array.from(new Set([
    ...GATEWAY_MESSAGE_PROVIDERS,
    "last",
    ...GATEWAY_AGENT_PROVIDER_ALIASES,
]));
export function isGatewayMessageProvider(value) {
    return GATEWAY_MESSAGE_PROVIDERS.includes(value);
}
export function isDeliverableMessageProvider(value) {
    return DELIVERABLE_MESSAGE_PROVIDERS.includes(value);
}
export function resolveGatewayMessageProvider(raw) {
    const normalized = normalizeMessageProvider(raw);
    if (!normalized)
        return undefined;
    return isGatewayMessageProvider(normalized) ? normalized : undefined;
}
export function resolveMessageProvider(primary, fallback) {
    return (normalizeMessageProvider(primary) ?? normalizeMessageProvider(fallback));
}
