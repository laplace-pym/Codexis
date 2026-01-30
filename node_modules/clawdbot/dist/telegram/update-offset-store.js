import crypto from "node:crypto";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { resolveStateDir } from "../config/paths.js";
const STORE_VERSION = 1;
function normalizeAccountId(accountId) {
    const trimmed = accountId?.trim();
    if (!trimmed)
        return "default";
    return trimmed.replace(/[^a-z0-9._-]+/gi, "_");
}
function resolveTelegramUpdateOffsetPath(accountId, env = process.env) {
    const stateDir = resolveStateDir(env, os.homedir);
    const normalized = normalizeAccountId(accountId);
    return path.join(stateDir, "telegram", `update-offset-${normalized}.json`);
}
function safeParseState(raw) {
    try {
        const parsed = JSON.parse(raw);
        if (parsed?.version !== STORE_VERSION)
            return null;
        if (parsed.lastUpdateId !== null && typeof parsed.lastUpdateId !== "number") {
            return null;
        }
        return parsed;
    }
    catch {
        return null;
    }
}
export async function readTelegramUpdateOffset(params) {
    const filePath = resolveTelegramUpdateOffsetPath(params.accountId, params.env);
    try {
        const raw = await fs.readFile(filePath, "utf-8");
        const parsed = safeParseState(raw);
        return parsed?.lastUpdateId ?? null;
    }
    catch (err) {
        const code = err.code;
        if (code === "ENOENT")
            return null;
        return null;
    }
}
export async function writeTelegramUpdateOffset(params) {
    const filePath = resolveTelegramUpdateOffsetPath(params.accountId, params.env);
    const dir = path.dirname(filePath);
    await fs.mkdir(dir, { recursive: true, mode: 0o700 });
    const tmp = path.join(dir, `${path.basename(filePath)}.${crypto.randomUUID()}.tmp`);
    const payload = {
        version: STORE_VERSION,
        lastUpdateId: params.updateId,
    };
    await fs.writeFile(tmp, `${JSON.stringify(payload, null, 2)}\n`, {
        encoding: "utf-8",
    });
    await fs.chmod(tmp, 0o600);
    await fs.rename(tmp, filePath);
}
