import { loadConfig } from "../../config/config.js";
import { handleDiscordAction } from "./discord-actions.js";
import { DiscordToolSchema } from "./discord-schema.js";
export function createDiscordTool() {
    return {
        label: "Discord",
        name: "discord",
        description: "Manage Discord messages, reactions, and moderation.",
        parameters: DiscordToolSchema,
        execute: async (_toolCallId, args) => {
            const params = args;
            const cfg = loadConfig();
            return await handleDiscordAction(params, cfg);
        },
    };
}
