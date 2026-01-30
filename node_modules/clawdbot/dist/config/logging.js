import { displayPath } from "../utils.js";
import { CONFIG_PATH_CLAWDBOT } from "./paths.js";
export function formatConfigPath(path = CONFIG_PATH_CLAWDBOT) {
    return displayPath(path);
}
export function logConfigUpdated(runtime, opts = {}) {
    const path = formatConfigPath(opts.path ?? CONFIG_PATH_CLAWDBOT);
    const suffix = opts.suffix ? ` ${opts.suffix}` : "";
    runtime.log(`Updated ${path}${suffix}`);
}
