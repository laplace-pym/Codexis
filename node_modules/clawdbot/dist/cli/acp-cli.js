import { runAcpClientInteractive } from "../acp/client.js";
import { serveAcpGateway } from "../acp/server.js";
import { defaultRuntime } from "../runtime.js";
import { formatDocsLink } from "../terminal/links.js";
import { theme } from "../terminal/theme.js";
export function registerAcpCli(program) {
    const acp = program.command("acp").description("Run an ACP bridge backed by the Gateway");
    acp
        .option("--url <url>", "Gateway WebSocket URL (defaults to gateway.remote.url when configured)")
        .option("--token <token>", "Gateway token (if required)")
        .option("--password <password>", "Gateway password (if required)")
        .option("--session <key>", "Default session key (e.g. agent:main:main)")
        .option("--session-label <label>", "Default session label to resolve")
        .option("--require-existing", "Fail if the session key/label does not exist", false)
        .option("--reset-session", "Reset the session key before first use", false)
        .option("--no-prefix-cwd", "Do not prefix prompts with the working directory", false)
        .option("--verbose, -v", "Verbose logging to stderr", false)
        .addHelpText("after", () => `\n${theme.muted("Docs:")} ${formatDocsLink("/cli/acp", "docs.clawd.bot/cli/acp")}\n`)
        .action((opts) => {
        try {
            serveAcpGateway({
                gatewayUrl: opts.url,
                gatewayToken: opts.token,
                gatewayPassword: opts.password,
                defaultSessionKey: opts.session,
                defaultSessionLabel: opts.sessionLabel,
                requireExistingSession: Boolean(opts.requireExisting),
                resetSession: Boolean(opts.resetSession),
                prefixCwd: !opts.noPrefixCwd,
                verbose: Boolean(opts.verbose),
            });
        }
        catch (err) {
            defaultRuntime.error(String(err));
            defaultRuntime.exit(1);
        }
    });
    acp
        .command("client")
        .description("Run an interactive ACP client against the local ACP bridge")
        .option("--cwd <dir>", "Working directory for the ACP session")
        .option("--server <command>", "ACP server command (default: clawdbot)")
        .option("--server-args <args...>", "Extra arguments for the ACP server")
        .option("--server-verbose", "Enable verbose logging on the ACP server", false)
        .option("--verbose, -v", "Verbose client logging", false)
        .action(async (opts) => {
        try {
            await runAcpClientInteractive({
                cwd: opts.cwd,
                serverCommand: opts.server,
                serverArgs: opts.serverArgs,
                serverVerbose: Boolean(opts.serverVerbose),
                verbose: Boolean(opts.verbose),
            });
        }
        catch (err) {
            defaultRuntime.error(String(err));
            defaultRuntime.exit(1);
        }
    });
}
