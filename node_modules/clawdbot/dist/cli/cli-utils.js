export function formatErrorMessage(err) {
    return err instanceof Error ? err.message : String(err);
}
export async function withManager(params) {
    const { manager, error } = await params.getManager();
    if (!manager) {
        params.onMissing(error);
        return;
    }
    try {
        await params.run(manager);
    }
    finally {
        try {
            await params.close(manager);
        }
        catch (err) {
            params.onCloseError?.(err);
        }
    }
}
export async function runCommandWithRuntime(runtime, action, onError) {
    try {
        await action();
    }
    catch (err) {
        if (onError) {
            onError(err);
            return;
        }
        runtime.error(String(err));
        runtime.exit(1);
    }
}
