export function buildChannelConfigSchema(schema) {
    return {
        schema: schema.toJSONSchema({
            target: "draft-07",
            unrepresentable: "any",
        }),
    };
}
