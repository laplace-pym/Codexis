import { requireActivePluginRegistry } from "./runtime.js";
import { normalizePluginHttpPath } from "./http-path.js";
export function registerPluginHttpRoute(params) {
    const registry = params.registry ?? requireActivePluginRegistry();
    const routes = registry.httpRoutes ?? [];
    registry.httpRoutes = routes;
    const normalizedPath = normalizePluginHttpPath(params.path, params.fallbackPath);
    const suffix = params.accountId ? ` for account "${params.accountId}"` : "";
    if (!normalizedPath) {
        params.log?.(`plugin: webhook path missing${suffix}`);
        return () => { };
    }
    if (routes.some((entry) => entry.path === normalizedPath)) {
        const pluginHint = params.pluginId ? ` (${params.pluginId})` : "";
        params.log?.(`plugin: webhook path ${normalizedPath} already registered${suffix}${pluginHint}`);
        return () => { };
    }
    const entry = {
        path: normalizedPath,
        handler: params.handler,
        pluginId: params.pluginId,
        source: params.source,
    };
    routes.push(entry);
    return () => {
        const index = routes.indexOf(entry);
        if (index >= 0) {
            routes.splice(index, 1);
        }
    };
}
