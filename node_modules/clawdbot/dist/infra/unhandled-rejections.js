import process from "node:process";
import { formatUncaughtError } from "./errors.js";
const handlers = new Set();
export function registerUnhandledRejectionHandler(handler) {
    handlers.add(handler);
    return () => {
        handlers.delete(handler);
    };
}
export function isUnhandledRejectionHandled(reason) {
    for (const handler of handlers) {
        try {
            if (handler(reason))
                return true;
        }
        catch (err) {
            console.error("[clawdbot] Unhandled rejection handler failed:", err instanceof Error ? (err.stack ?? err.message) : err);
        }
    }
    return false;
}
export function installUnhandledRejectionHandler() {
    process.on("unhandledRejection", (reason, _promise) => {
        if (isUnhandledRejectionHandled(reason))
            return;
        console.error("[clawdbot] Unhandled promise rejection:", formatUncaughtError(reason));
        process.exit(1);
    });
}
