export async function getMemorySearchManager(params) {
    try {
        const { MemoryIndexManager } = await import("./manager.js");
        const manager = await MemoryIndexManager.get(params);
        return { manager };
    }
    catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        return { manager: null, error: message };
    }
}
