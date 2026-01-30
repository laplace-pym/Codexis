import { applyAuthChoicePluginProvider } from "./auth-choice.apply.plugin-provider.js";
export async function applyAuthChoiceGoogleGeminiCli(params) {
    return await applyAuthChoicePluginProvider(params, {
        authChoice: "google-gemini-cli",
        pluginId: "google-gemini-cli-auth",
        providerId: "google-gemini-cli",
        methodId: "oauth",
        label: "Google Gemini CLI",
    });
}
