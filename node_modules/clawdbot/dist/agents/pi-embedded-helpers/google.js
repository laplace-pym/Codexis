import { sanitizeGoogleTurnOrdering } from "./bootstrap.js";
export function isGoogleModelApi(api) {
    return (api === "google-gemini-cli" || api === "google-generative-ai" || api === "google-antigravity");
}
export function isAntigravityClaude(params) {
    const provider = params.provider?.toLowerCase();
    const api = params.api?.toLowerCase();
    if (provider !== "google-antigravity" && api !== "google-antigravity")
        return false;
    return params.modelId?.toLowerCase().includes("claude") ?? false;
}
export { sanitizeGoogleTurnOrdering };
