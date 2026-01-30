import { messageCommand } from "../../../commands/message.js";
import { danger, setVerbose } from "../../../globals.js";
import { CHANNEL_TARGET_DESCRIPTION } from "../../../infra/outbound/channel-target.js";
import { defaultRuntime } from "../../../runtime.js";
import { createDefaultDeps } from "../../deps.js";
import { runCommandWithRuntime } from "../../cli-utils.js";
import { ensurePluginRegistryLoaded } from "../../plugin-registry.js";
export function createMessageCliHelpers(message, messageChannelOptions) {
    const withMessageBase = (command) => command
        .option("--channel <channel>", `Channel: ${messageChannelOptions}`)
        .option("--account <id>", "Channel account id (accountId)")
        .option("--json", "Output result as JSON", false)
        .option("--dry-run", "Print payload and skip sending", false)
        .option("--verbose", "Verbose logging", false);
    const withMessageTarget = (command) => command.option("-t, --target <dest>", CHANNEL_TARGET_DESCRIPTION);
    const withRequiredMessageTarget = (command) => command.requiredOption("-t, --target <dest>", CHANNEL_TARGET_DESCRIPTION);
    const runMessageAction = async (action, opts) => {
        setVerbose(Boolean(opts.verbose));
        ensurePluginRegistryLoaded();
        const deps = createDefaultDeps();
        await runCommandWithRuntime(defaultRuntime, async () => {
            await messageCommand({
                ...(() => {
                    const { account, ...rest } = opts;
                    return {
                        ...rest,
                        accountId: typeof account === "string" ? account : undefined,
                    };
                })(),
                action,
            }, deps, defaultRuntime);
        }, (err) => {
            defaultRuntime.error(danger(String(err)));
            defaultRuntime.exit(1);
        });
    };
    // `message` is only used for `message.help({ error: true })`, keep the
    // command-specific helpers grouped here.
    void message;
    return {
        withMessageBase,
        withMessageTarget,
        withRequiredMessageTarget,
        runMessageAction,
    };
}
