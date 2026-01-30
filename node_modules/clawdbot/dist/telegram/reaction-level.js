import { resolveTelegramAccount } from "./accounts.js";
/**
 * Resolve the effective reaction level and its implications.
 */
export function resolveTelegramReactionLevel(params) {
    const account = resolveTelegramAccount({
        cfg: params.cfg,
        accountId: params.accountId,
    });
    const level = (account.config.reactionLevel ?? "minimal");
    switch (level) {
        case "off":
            return {
                level,
                ackEnabled: false,
                agentReactionsEnabled: false,
            };
        case "ack":
            return {
                level,
                ackEnabled: true,
                agentReactionsEnabled: false,
            };
        case "minimal":
            return {
                level,
                ackEnabled: false,
                agentReactionsEnabled: true,
                agentReactionGuidance: "minimal",
            };
        case "extensive":
            return {
                level,
                ackEnabled: false,
                agentReactionsEnabled: true,
                agentReactionGuidance: "extensive",
            };
        default:
            // Fallback to ack behavior
            return {
                level: "ack",
                ackEnabled: true,
                agentReactionsEnabled: false,
            };
    }
}
