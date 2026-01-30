export const DEFAULT_GITHUB_COPILOT_BASE_URL = "https://api.githubcopilot.com";
export function resolveGithubCopilotUserAgent() {
    const version = process.env.CLAWDBOT_VERSION ?? process.env.npm_package_version ?? "unknown";
    return `clawdbot/${version}`;
}
export function normalizeGithubCopilotDomain(input) {
    const trimmed = (input ?? "").trim();
    if (!trimmed)
        return null;
    try {
        const url = trimmed.includes("://") ? new URL(trimmed) : new URL(`https://${trimmed}`);
        return url.hostname;
    }
    catch {
        return null;
    }
}
export function resolveGithubCopilotBaseUrl(enterpriseDomain) {
    if (enterpriseDomain && enterpriseDomain.trim()) {
        return `https://copilot-api.${enterpriseDomain.trim()}`;
    }
    return DEFAULT_GITHUB_COPILOT_BASE_URL;
}
