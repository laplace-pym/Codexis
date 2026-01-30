export function registerMessagePinCommands(message, helpers) {
    const pins = [
        helpers
            .withMessageBase(helpers.withRequiredMessageTarget(message.command("pin").description("Pin a message")))
            .requiredOption("--message-id <id>", "Message id")
            .action(async (opts) => {
            await helpers.runMessageAction("pin", opts);
        }),
        helpers
            .withMessageBase(helpers.withRequiredMessageTarget(message.command("unpin").description("Unpin a message")))
            .requiredOption("--message-id <id>", "Message id")
            .action(async (opts) => {
            await helpers.runMessageAction("unpin", opts);
        }),
        helpers
            .withMessageBase(helpers.withRequiredMessageTarget(message.command("pins").description("List pinned messages")))
            .option("--limit <n>", "Result limit")
            .action(async (opts) => {
            await helpers.runMessageAction("list-pins", opts);
        }),
    ];
    void pins;
}
