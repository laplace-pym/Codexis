const createEmptyRegistry = () => ({
    plugins: [],
    tools: [],
    hooks: [],
    typedHooks: [],
    channels: [],
    providers: [],
    gatewayHandlers: {},
    httpHandlers: [],
    httpRoutes: [],
    cliRegistrars: [],
    services: [],
    commands: [],
    diagnostics: [],
});
const REGISTRY_STATE = Symbol.for("clawdbot.pluginRegistryState");
const state = (() => {
    const globalState = globalThis;
    if (!globalState[REGISTRY_STATE]) {
        globalState[REGISTRY_STATE] = {
            registry: createEmptyRegistry(),
            key: null,
        };
    }
    return globalState[REGISTRY_STATE];
})();
export function setActivePluginRegistry(registry, cacheKey) {
    state.registry = registry;
    state.key = cacheKey ?? null;
}
export function getActivePluginRegistry() {
    return state.registry;
}
export function requireActivePluginRegistry() {
    if (!state.registry) {
        state.registry = createEmptyRegistry();
    }
    return state.registry;
}
export function getActivePluginRegistryKey() {
    return state.key;
}
