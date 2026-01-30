import { listProviderPlugins } from "../providers/plugins/index.js";
export function listProviderAgentTools(params) {
    // Provider docking: aggregate provider-owned tools (login, etc.).
    const tools = [];
    for (const plugin of listProviderPlugins()) {
        const entry = plugin.agentTools;
        if (!entry)
            continue;
        const resolved = typeof entry === "function" ? entry(params) : entry;
        if (Array.isArray(resolved))
            tools.push(...resolved);
    }
    return tools;
}
