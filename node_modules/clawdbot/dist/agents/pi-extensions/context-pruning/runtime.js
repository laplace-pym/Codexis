// Session-scoped runtime registry keyed by object identity.
// Important: this relies on Pi passing the same SessionManager object instance into
// ExtensionContext (ctx.sessionManager) that we used when calling setContextPruningRuntime.
const REGISTRY = new WeakMap();
export function setContextPruningRuntime(sessionManager, value) {
    if (!sessionManager || typeof sessionManager !== "object") {
        return;
    }
    const key = sessionManager;
    if (value === null) {
        REGISTRY.delete(key);
        return;
    }
    REGISTRY.set(key, value);
}
export function getContextPruningRuntime(sessionManager) {
    if (!sessionManager || typeof sessionManager !== "object") {
        return null;
    }
    return REGISTRY.get(sessionManager) ?? null;
}
