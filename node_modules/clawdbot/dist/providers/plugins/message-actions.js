import { getProviderPlugin, listProviderPlugins } from "./index.js";
export function listProviderMessageActions(cfg) {
    const actions = new Set(["send"]);
    for (const plugin of listProviderPlugins()) {
        const list = plugin.actions?.listActions?.({ cfg });
        if (!list)
            continue;
        for (const action of list)
            actions.add(action);
    }
    return Array.from(actions);
}
export function supportsProviderMessageButtons(cfg) {
    for (const plugin of listProviderPlugins()) {
        if (plugin.actions?.supportsButtons?.({ cfg }))
            return true;
    }
    return false;
}
export async function dispatchProviderMessageAction(ctx) {
    const plugin = getProviderPlugin(ctx.provider);
    if (!plugin?.actions?.handleAction)
        return null;
    if (plugin.actions.supportsAction &&
        !plugin.actions.supportsAction({ action: ctx.action })) {
        return null;
    }
    return await plugin.actions.handleAction(ctx);
}
