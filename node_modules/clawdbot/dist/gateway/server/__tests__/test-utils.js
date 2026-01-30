export const createTestRegistry = (overrides = {}) => {
    const base = {
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
    };
    const merged = { ...base, ...overrides };
    return {
        ...merged,
        gatewayHandlers: merged.gatewayHandlers ?? {},
        httpHandlers: merged.httpHandlers ?? [],
        httpRoutes: merged.httpRoutes ?? [],
    };
};
