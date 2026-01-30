function error(message) {
    return { success: false, error: { issues: [{ path: [], message }] } };
}
export function emptyPluginConfigSchema() {
    return {
        safeParse(value) {
            if (value === undefined)
                return { success: true, data: undefined };
            if (!value || typeof value !== "object" || Array.isArray(value)) {
                return error("expected config object");
            }
            if (Object.keys(value).length > 0) {
                return error("config must be empty");
            }
            return { success: true, data: value };
        },
        jsonSchema: {
            type: "object",
            additionalProperties: false,
            properties: {},
        },
    };
}
