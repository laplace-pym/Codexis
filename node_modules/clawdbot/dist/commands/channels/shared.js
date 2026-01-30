import { getChannelPlugin } from "../../channels/plugins/index.js";
import { formatCliCommand } from "../../cli/command-format.js";
import { readConfigFileSnapshot } from "../../config/config.js";
import { DEFAULT_ACCOUNT_ID } from "../../routing/session-key.js";
import { defaultRuntime } from "../../runtime.js";
export async function requireValidConfig(runtime = defaultRuntime) {
    const snapshot = await readConfigFileSnapshot();
    if (snapshot.exists && !snapshot.valid) {
        const issues = snapshot.issues.length > 0
            ? snapshot.issues.map((issue) => `- ${issue.path}: ${issue.message}`).join("\n")
            : "Unknown validation issue.";
        runtime.error(`Config invalid:\n${issues}`);
        runtime.error(`Fix the config or run ${formatCliCommand("clawdbot doctor")}.`);
        runtime.exit(1);
        return null;
    }
    return snapshot.config;
}
export function formatAccountLabel(params) {
    const base = params.accountId || DEFAULT_ACCOUNT_ID;
    if (params.name?.trim())
        return `${base} (${params.name.trim()})`;
    return base;
}
export const channelLabel = (channel) => {
    const plugin = getChannelPlugin(channel);
    return plugin?.meta.label ?? channel;
};
export function formatChannelAccountLabel(params) {
    const channelText = channelLabel(params.channel);
    const accountText = formatAccountLabel({
        accountId: params.accountId,
        name: params.name,
    });
    const styledChannel = params.channelStyle ? params.channelStyle(channelText) : channelText;
    const styledAccount = params.accountStyle ? params.accountStyle(accountText) : accountText;
    return `${styledChannel} ${styledAccount}`;
}
export function shouldUseWizard(params) {
    return params?.hasFlags === false;
}
