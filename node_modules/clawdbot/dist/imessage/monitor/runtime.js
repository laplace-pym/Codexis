export function resolveRuntime(opts) {
    return (opts.runtime ?? {
        log: console.log,
        error: console.error,
        exit: (code) => {
            throw new Error(`exit ${code}`);
        },
    });
}
export function normalizeAllowList(list) {
    return (list ?? []).map((entry) => String(entry).trim()).filter(Boolean);
}
