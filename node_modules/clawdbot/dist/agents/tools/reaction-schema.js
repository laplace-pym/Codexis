import { Type } from "@sinclair/typebox";
export function createReactionSchema(options) {
    const schema = {
        action: Type.Literal(options.action ?? "react"),
        ...options.ids,
        emoji: options.emoji ?? Type.String(),
    };
    if (options.includeRemove) {
        schema.remove = Type.Optional(Type.Boolean());
    }
    if (options.extras) {
        Object.assign(schema, options.extras);
    }
    return Type.Object(schema);
}
