import { getActivePluginRegistry } from "../../../plugins/runtime.js";
// Channel docking: outbound sends should stay cheap to import.
//
// The full channel plugins (src/channels/plugins/*.ts) pull in status,
// onboarding, gateway monitors, etc. Outbound delivery only needs chunking +
// send primitives, so we keep a dedicated, lightweight loader here.
const cache = new Map();
let lastRegistry = null;
function ensureCacheForRegistry(registry) {
    if (registry === lastRegistry)
        return;
    cache.clear();
    lastRegistry = registry;
}
export async function loadChannelOutboundAdapter(id) {
    const registry = getActivePluginRegistry();
    ensureCacheForRegistry(registry);
    const cached = cache.get(id);
    if (cached)
        return cached;
    const pluginEntry = registry?.channels.find((entry) => entry.plugin.id === id);
    const outbound = pluginEntry?.plugin.outbound;
    if (outbound) {
        cache.set(id, outbound);
        return outbound;
    }
    return undefined;
}
