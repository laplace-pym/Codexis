import { loadConfig } from "../../config/config.js";
import { handleTelegramAction } from "./telegram-actions.js";
import { TelegramToolSchema } from "./telegram-schema.js";
export function createTelegramTool() {
    return {
        label: "Telegram",
        name: "telegram",
        description: "Send messages and manage reactions on Telegram.",
        parameters: TelegramToolSchema,
        execute: async (_toolCallId, args) => {
            const params = args;
            const cfg = loadConfig();
            return await handleTelegramAction(params, cfg);
        },
    };
}
