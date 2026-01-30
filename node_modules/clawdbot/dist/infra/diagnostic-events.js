let seq = 0;
const listeners = new Set();
export function isDiagnosticsEnabled(config) {
    return config?.diagnostics?.enabled === true;
}
export function emitDiagnosticEvent(event) {
    const enriched = {
        ...event,
        seq: (seq += 1),
        ts: Date.now(),
    };
    for (const listener of listeners) {
        try {
            listener(enriched);
        }
        catch {
            // Ignore listener failures.
        }
    }
}
export function onDiagnosticEvent(listener) {
    listeners.add(listener);
    return () => listeners.delete(listener);
}
export function resetDiagnosticEventsForTest() {
    seq = 0;
    listeners.clear();
}
