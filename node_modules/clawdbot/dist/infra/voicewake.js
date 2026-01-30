import { randomUUID } from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import { resolveStateDir } from "../config/paths.js";
const DEFAULT_TRIGGERS = ["clawd", "claude", "computer"];
function resolvePath(baseDir) {
    const root = baseDir ?? resolveStateDir();
    return path.join(root, "settings", "voicewake.json");
}
function sanitizeTriggers(triggers) {
    const cleaned = (triggers ?? [])
        .map((w) => (typeof w === "string" ? w.trim() : ""))
        .filter((w) => w.length > 0);
    return cleaned.length > 0 ? cleaned : DEFAULT_TRIGGERS;
}
async function readJSON(filePath) {
    try {
        const raw = await fs.readFile(filePath, "utf8");
        return JSON.parse(raw);
    }
    catch {
        return null;
    }
}
async function writeJSONAtomic(filePath, value) {
    const dir = path.dirname(filePath);
    await fs.mkdir(dir, { recursive: true });
    const tmp = `${filePath}.${randomUUID()}.tmp`;
    await fs.writeFile(tmp, JSON.stringify(value, null, 2), "utf8");
    await fs.rename(tmp, filePath);
}
let lock = Promise.resolve();
async function withLock(fn) {
    const prev = lock;
    let release;
    lock = new Promise((resolve) => {
        release = resolve;
    });
    await prev;
    try {
        return await fn();
    }
    finally {
        release?.();
    }
}
export function defaultVoiceWakeTriggers() {
    return [...DEFAULT_TRIGGERS];
}
export async function loadVoiceWakeConfig(baseDir) {
    const filePath = resolvePath(baseDir);
    const existing = await readJSON(filePath);
    if (!existing) {
        return { triggers: defaultVoiceWakeTriggers(), updatedAtMs: 0 };
    }
    return {
        triggers: sanitizeTriggers(existing.triggers),
        updatedAtMs: typeof existing.updatedAtMs === "number" && existing.updatedAtMs > 0
            ? existing.updatedAtMs
            : 0,
    };
}
export async function setVoiceWakeTriggers(triggers, baseDir) {
    const sanitized = sanitizeTriggers(triggers);
    const filePath = resolvePath(baseDir);
    return await withLock(async () => {
        const next = {
            triggers: sanitized,
            updatedAtMs: Date.now(),
        };
        await writeJSONAtomic(filePath, next);
        return next;
    });
}
