import { formatCliCommand } from "../cli/command-format.js";
import { readConfigFileSnapshot } from "../config/config.js";
import { defaultRuntime } from "../runtime.js";
import { runNonInteractiveOnboardingLocal } from "./onboard-non-interactive/local.js";
import { runNonInteractiveOnboardingRemote } from "./onboard-non-interactive/remote.js";
export async function runNonInteractiveOnboarding(opts, runtime = defaultRuntime) {
    const snapshot = await readConfigFileSnapshot();
    if (snapshot.exists && !snapshot.valid) {
        runtime.error(`Config invalid. Run \`${formatCliCommand("clawdbot doctor")}\` to repair it, then re-run onboarding.`);
        runtime.exit(1);
        return;
    }
    const baseConfig = snapshot.valid ? snapshot.config : {};
    const mode = opts.mode ?? "local";
    if (mode !== "local" && mode !== "remote") {
        runtime.error(`Invalid --mode "${String(mode)}" (use local|remote).`);
        runtime.exit(1);
        return;
    }
    if (mode === "remote") {
        await runNonInteractiveOnboardingRemote({ opts, runtime, baseConfig });
        return;
    }
    await runNonInteractiveOnboardingLocal({ opts, runtime, baseConfig });
}
