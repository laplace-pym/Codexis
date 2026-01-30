import { createSubsystemLogger } from "../logging/subsystem.js";
import { buildFileEntry, listMemoryFiles } from "./internal.js";
const log = createSubsystemLogger("memory");
export async function syncMemoryFiles(params) {
    const files = await listMemoryFiles(params.workspaceDir);
    const fileEntries = await Promise.all(files.map(async (file) => buildFileEntry(file, params.workspaceDir)));
    log.debug("memory sync: indexing memory files", {
        files: fileEntries.length,
        needsFullReindex: params.needsFullReindex,
        batch: params.batchEnabled,
        concurrency: params.concurrency,
    });
    const activePaths = new Set(fileEntries.map((entry) => entry.path));
    if (params.progress) {
        params.progress.total += fileEntries.length;
        params.progress.report({
            completed: params.progress.completed,
            total: params.progress.total,
            label: params.batchEnabled ? "Indexing memory files (batch)..." : "Indexing memory files…",
        });
    }
    const tasks = fileEntries.map((entry) => async () => {
        const record = params.db
            .prepare(`SELECT hash FROM files WHERE path = ? AND source = ?`)
            .get(entry.path, "memory");
        if (!params.needsFullReindex && record?.hash === entry.hash) {
            if (params.progress) {
                params.progress.completed += 1;
                params.progress.report({
                    completed: params.progress.completed,
                    total: params.progress.total,
                });
            }
            return;
        }
        await params.indexFile(entry);
        if (params.progress) {
            params.progress.completed += 1;
            params.progress.report({
                completed: params.progress.completed,
                total: params.progress.total,
            });
        }
    });
    await params.runWithConcurrency(tasks, params.concurrency);
    const staleRows = params.db
        .prepare(`SELECT path FROM files WHERE source = ?`)
        .all("memory");
    for (const stale of staleRows) {
        if (activePaths.has(stale.path))
            continue;
        params.db.prepare(`DELETE FROM files WHERE path = ? AND source = ?`).run(stale.path, "memory");
        try {
            params.db
                .prepare(`DELETE FROM ${params.vectorTable} WHERE id IN (SELECT id FROM chunks WHERE path = ? AND source = ?)`)
                .run(stale.path, "memory");
        }
        catch { }
        params.db.prepare(`DELETE FROM chunks WHERE path = ? AND source = ?`).run(stale.path, "memory");
        if (params.ftsEnabled && params.ftsAvailable) {
            try {
                params.db
                    .prepare(`DELETE FROM ${params.ftsTable} WHERE path = ? AND source = ? AND model = ?`)
                    .run(stale.path, "memory", params.model);
            }
            catch { }
        }
    }
}
