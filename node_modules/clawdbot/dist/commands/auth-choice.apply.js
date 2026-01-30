import { applyAuthChoiceAnthropic } from "./auth-choice.apply.anthropic.js";
import { applyAuthChoiceApiProviders } from "./auth-choice.apply.api-providers.js";
import { applyAuthChoiceCopilotProxy } from "./auth-choice.apply.copilot-proxy.js";
import { applyAuthChoiceGitHubCopilot } from "./auth-choice.apply.github-copilot.js";
import { applyAuthChoiceGoogleAntigravity } from "./auth-choice.apply.google-antigravity.js";
import { applyAuthChoiceGoogleGeminiCli } from "./auth-choice.apply.google-gemini-cli.js";
import { applyAuthChoiceMiniMax } from "./auth-choice.apply.minimax.js";
import { applyAuthChoiceOAuth } from "./auth-choice.apply.oauth.js";
import { applyAuthChoiceOpenAI } from "./auth-choice.apply.openai.js";
import { applyAuthChoiceQwenPortal } from "./auth-choice.apply.qwen-portal.js";
export async function applyAuthChoice(params) {
    const handlers = [
        applyAuthChoiceAnthropic,
        applyAuthChoiceOpenAI,
        applyAuthChoiceOAuth,
        applyAuthChoiceApiProviders,
        applyAuthChoiceMiniMax,
        applyAuthChoiceGitHubCopilot,
        applyAuthChoiceGoogleAntigravity,
        applyAuthChoiceGoogleGeminiCli,
        applyAuthChoiceCopilotProxy,
        applyAuthChoiceQwenPortal,
    ];
    for (const handler of handlers) {
        const result = await handler(params);
        if (result)
            return result;
    }
    return { config: params.config };
}
