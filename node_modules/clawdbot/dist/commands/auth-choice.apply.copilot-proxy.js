import { applyAuthChoicePluginProvider } from "./auth-choice.apply.plugin-provider.js";
export async function applyAuthChoiceCopilotProxy(params) {
    return await applyAuthChoicePluginProvider(params, {
        authChoice: "copilot-proxy",
        pluginId: "copilot-proxy",
        providerId: "copilot-proxy",
        methodId: "local",
        label: "Copilot Proxy",
    });
}
