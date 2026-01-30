import crypto from "node:crypto";
export function mergeSessionEntry(existing, patch) {
    const sessionId = patch.sessionId ?? existing?.sessionId ?? crypto.randomUUID();
    const updatedAt = Math.max(existing?.updatedAt ?? 0, patch.updatedAt ?? 0, Date.now());
    if (!existing)
        return { ...patch, sessionId, updatedAt };
    return { ...existing, ...patch, sessionId, updatedAt };
}
export const DEFAULT_RESET_TRIGGER = "/new";
export const DEFAULT_RESET_TRIGGERS = ["/new", "/reset"];
export const DEFAULT_IDLE_MINUTES = 60;
