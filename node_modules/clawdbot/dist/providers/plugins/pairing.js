import { getProviderPlugin, listProviderPlugins, normalizeProviderId, } from "./index.js";
export function listPairingProviders() {
    // Provider docking: pairing support is declared via plugin.pairing.
    return listProviderPlugins()
        .filter((plugin) => plugin.pairing)
        .map((plugin) => plugin.id);
}
export function getPairingAdapter(providerId) {
    const plugin = getProviderPlugin(providerId);
    return plugin?.pairing ?? null;
}
export function requirePairingAdapter(providerId) {
    const adapter = getPairingAdapter(providerId);
    if (!adapter) {
        throw new Error(`Provider ${providerId} does not support pairing`);
    }
    return adapter;
}
export function resolvePairingProvider(raw) {
    const value = (typeof raw === "string"
        ? raw
        : typeof raw === "number" || typeof raw === "boolean"
            ? String(raw)
            : "")
        .trim()
        .toLowerCase();
    const normalized = normalizeProviderId(value);
    const providers = listPairingProviders();
    if (!normalized || !providers.includes(normalized)) {
        throw new Error(`Invalid provider: ${value || "(empty)"} (expected one of: ${providers.join(", ")})`);
    }
    return normalized;
}
export async function notifyPairingApproved(params) {
    const adapter = requirePairingAdapter(params.providerId);
    if (!adapter.notifyApproval)
        return;
    await adapter.notifyApproval({
        cfg: params.cfg,
        id: params.id,
        runtime: params.runtime,
    });
}
