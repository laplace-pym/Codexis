import { danger } from "../../globals.js";
import { defaultRuntime } from "../../runtime.js";
import { formatDocsLink } from "../../terminal/links.js";
import { theme } from "../../terminal/theme.js";
import { addGatewayClientOptions, callGatewayFromCli } from "../gateway-rpc.js";
export function registerWakeCommand(program) {
    addGatewayClientOptions(program
        .command("wake")
        .description("Enqueue a system event and optionally trigger an immediate heartbeat")
        .requiredOption("--text <text>", "System event text")
        .option("--mode <mode>", "Wake mode (now|next-heartbeat)", "next-heartbeat")
        .option("--json", "Output JSON", false))
        .addHelpText("after", () => `\n${theme.muted("Docs:")} ${formatDocsLink("/cli/wake", "docs.clawd.bot/cli/wake")}\n`)
        .action(async (opts) => {
        try {
            const result = await callGatewayFromCli("wake", opts, { mode: opts.mode, text: opts.text }, { expectFinal: false });
            if (opts.json)
                defaultRuntime.log(JSON.stringify(result, null, 2));
            else
                defaultRuntime.log("ok");
        }
        catch (err) {
            defaultRuntime.error(danger(String(err)));
            defaultRuntime.exit(1);
        }
    });
}
