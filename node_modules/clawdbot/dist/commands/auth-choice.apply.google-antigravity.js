import { applyAuthChoicePluginProvider } from "./auth-choice.apply.plugin-provider.js";
export async function applyAuthChoiceGoogleAntigravity(params) {
    return await applyAuthChoicePluginProvider(params, {
        authChoice: "google-antigravity",
        pluginId: "google-antigravity-auth",
        providerId: "google-antigravity",
        methodId: "oauth",
        label: "Google Antigravity",
    });
}
