import { writeConfigFile } from "../../config/config.js";
import { logConfigUpdated } from "../../config/logging.js";
import { formatCliCommand } from "../../cli/command-format.js";
import { applyWizardMetadata } from "../onboard-helpers.js";
export async function runNonInteractiveOnboardingRemote(params) {
    const { opts, runtime, baseConfig } = params;
    const mode = "remote";
    const remoteUrl = opts.remoteUrl?.trim();
    if (!remoteUrl) {
        runtime.error("Missing --remote-url for remote mode.");
        runtime.exit(1);
        return;
    }
    let nextConfig = {
        ...baseConfig,
        gateway: {
            ...baseConfig.gateway,
            mode: "remote",
            remote: {
                url: remoteUrl,
                token: opts.remoteToken?.trim() || undefined,
            },
        },
    };
    nextConfig = applyWizardMetadata(nextConfig, { command: "onboard", mode });
    await writeConfigFile(nextConfig);
    logConfigUpdated(runtime);
    const payload = {
        mode,
        remoteUrl,
        auth: opts.remoteToken ? "token" : "none",
    };
    if (opts.json) {
        runtime.log(JSON.stringify(payload, null, 2));
    }
    else {
        runtime.log(`Remote gateway: ${remoteUrl}`);
        runtime.log(`Auth: ${payload.auth}`);
        runtime.log(`Tip: run \`${formatCliCommand("clawdbot configure --section web")}\` to store your Brave API key for web_search. Docs: https://docs.clawd.bot/tools/web`);
    }
}
