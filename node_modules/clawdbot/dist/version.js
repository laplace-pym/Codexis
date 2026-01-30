import { createRequire } from "node:module";
function readVersionFromPackageJson() {
    try {
        const require = createRequire(import.meta.url);
        const pkg = require("../package.json");
        return pkg.version ?? null;
    }
    catch {
        return null;
    }
}
// Single source of truth for the current clawdbot version.
// - Embedded/bundled builds: injected define or env var.
// - Dev/npm builds: package.json.
export const VERSION = (typeof __CLAWDBOT_VERSION__ === "string" && __CLAWDBOT_VERSION__) ||
    process.env.CLAWDBOT_BUNDLED_VERSION ||
    readVersionFromPackageJson() ||
    "0.0.0";
