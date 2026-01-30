import fs from "node:fs/promises";
import { SANDBOX_BROWSER_REGISTRY_PATH, SANDBOX_REGISTRY_PATH, SANDBOX_STATE_DIR, } from "./constants.js";
export async function readRegistry() {
    try {
        const raw = await fs.readFile(SANDBOX_REGISTRY_PATH, "utf-8");
        const parsed = JSON.parse(raw);
        if (parsed && Array.isArray(parsed.entries))
            return parsed;
    }
    catch {
        // ignore
    }
    return { entries: [] };
}
async function writeRegistry(registry) {
    await fs.mkdir(SANDBOX_STATE_DIR, { recursive: true });
    await fs.writeFile(SANDBOX_REGISTRY_PATH, `${JSON.stringify(registry, null, 2)}\n`, "utf-8");
}
export async function updateRegistry(entry) {
    const registry = await readRegistry();
    const existing = registry.entries.find((item) => item.containerName === entry.containerName);
    const next = registry.entries.filter((item) => item.containerName !== entry.containerName);
    next.push({
        ...entry,
        createdAtMs: existing?.createdAtMs ?? entry.createdAtMs,
        image: existing?.image ?? entry.image,
        configHash: entry.configHash ?? existing?.configHash,
    });
    await writeRegistry({ entries: next });
}
export async function removeRegistryEntry(containerName) {
    const registry = await readRegistry();
    const next = registry.entries.filter((item) => item.containerName !== containerName);
    if (next.length === registry.entries.length)
        return;
    await writeRegistry({ entries: next });
}
export async function readBrowserRegistry() {
    try {
        const raw = await fs.readFile(SANDBOX_BROWSER_REGISTRY_PATH, "utf-8");
        const parsed = JSON.parse(raw);
        if (parsed && Array.isArray(parsed.entries))
            return parsed;
    }
    catch {
        // ignore
    }
    return { entries: [] };
}
async function writeBrowserRegistry(registry) {
    await fs.mkdir(SANDBOX_STATE_DIR, { recursive: true });
    await fs.writeFile(SANDBOX_BROWSER_REGISTRY_PATH, `${JSON.stringify(registry, null, 2)}\n`, "utf-8");
}
export async function updateBrowserRegistry(entry) {
    const registry = await readBrowserRegistry();
    const existing = registry.entries.find((item) => item.containerName === entry.containerName);
    const next = registry.entries.filter((item) => item.containerName !== entry.containerName);
    next.push({
        ...entry,
        createdAtMs: existing?.createdAtMs ?? entry.createdAtMs,
        image: existing?.image ?? entry.image,
    });
    await writeBrowserRegistry({ entries: next });
}
export async function removeBrowserRegistryEntry(containerName) {
    const registry = await readBrowserRegistry();
    const next = registry.entries.filter((item) => item.containerName !== containerName);
    if (next.length === registry.entries.length)
        return;
    await writeBrowserRegistry({ entries: next });
}
