import { collectOption } from "../helpers.js";
export function registerMessagePermissionsCommand(message, helpers) {
    helpers
        .withMessageBase(helpers.withRequiredMessageTarget(message.command("permissions").description("Fetch channel permissions")))
        .action(async (opts) => {
        await helpers.runMessageAction("permissions", opts);
    });
}
export function registerMessageSearchCommand(message, helpers) {
    helpers
        .withMessageBase(message.command("search").description("Search Discord messages"))
        .requiredOption("--guild-id <id>", "Guild id")
        .requiredOption("--query <text>", "Search query")
        .option("--channel-id <id>", "Channel id")
        .option("--channel-ids <id>", "Channel id (repeat)", collectOption, [])
        .option("--author-id <id>", "Author id")
        .option("--author-ids <id>", "Author id (repeat)", collectOption, [])
        .option("--limit <n>", "Result limit")
        .action(async (opts) => {
        await helpers.runMessageAction("search", opts);
    });
}
