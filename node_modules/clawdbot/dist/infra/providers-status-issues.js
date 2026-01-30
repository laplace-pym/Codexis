import { listProviderPlugins } from "../providers/plugins/index.js";
export function collectProvidersStatusIssues(payload) {
    const issues = [];
    const accountsByProvider = payload.providerAccounts;
    for (const plugin of listProviderPlugins()) {
        const collect = plugin.status?.collectStatusIssues;
        if (!collect)
            continue;
        const raw = accountsByProvider?.[plugin.id];
        if (!Array.isArray(raw))
            continue;
        issues.push(...collect(raw));
    }
    return issues;
}
