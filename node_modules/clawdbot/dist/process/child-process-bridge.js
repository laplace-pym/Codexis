import process from "node:process";
const defaultSignals = process.platform === "win32"
    ? ["SIGTERM", "SIGINT", "SIGBREAK"]
    : ["SIGTERM", "SIGINT", "SIGHUP", "SIGQUIT"];
export function attachChildProcessBridge(child, { signals = defaultSignals, onSignal } = {}) {
    const listeners = new Map();
    for (const signal of signals) {
        const listener = () => {
            onSignal?.(signal);
            try {
                child.kill(signal);
            }
            catch {
                // ignore
            }
        };
        try {
            process.on(signal, listener);
            listeners.set(signal, listener);
        }
        catch {
            // Unsupported signal on this platform.
        }
    }
    const detach = () => {
        for (const [signal, listener] of listeners) {
            process.off(signal, listener);
        }
        listeners.clear();
    };
    child.once("exit", detach);
    child.once("error", detach);
    return { detach };
}
