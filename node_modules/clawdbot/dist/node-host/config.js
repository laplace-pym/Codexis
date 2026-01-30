import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import { resolveStateDir } from "../config/paths.js";
const NODE_HOST_FILE = "node.json";
export function resolveNodeHostConfigPath() {
    return path.join(resolveStateDir(), NODE_HOST_FILE);
}
function normalizeConfig(config) {
    const base = {
        version: 1,
        nodeId: "",
        token: config?.token,
        displayName: config?.displayName,
        gateway: config?.gateway,
    };
    if (config?.version === 1 && typeof config.nodeId === "string") {
        base.nodeId = config.nodeId.trim();
    }
    if (!base.nodeId) {
        base.nodeId = crypto.randomUUID();
    }
    return base;
}
export async function loadNodeHostConfig() {
    const filePath = resolveNodeHostConfigPath();
    try {
        const raw = await fs.readFile(filePath, "utf8");
        const parsed = JSON.parse(raw);
        return normalizeConfig(parsed);
    }
    catch {
        return null;
    }
}
export async function saveNodeHostConfig(config) {
    const filePath = resolveNodeHostConfigPath();
    await fs.mkdir(path.dirname(filePath), { recursive: true });
    const payload = JSON.stringify(config, null, 2);
    await fs.writeFile(filePath, `${payload}\n`, { mode: 0o600 });
    try {
        await fs.chmod(filePath, 0o600);
    }
    catch {
        // best-effort on platforms without chmod
    }
}
export async function ensureNodeHostConfig() {
    const existing = await loadNodeHostConfig();
    const normalized = normalizeConfig(existing);
    await saveNodeHostConfig(normalized);
    return normalized;
}
