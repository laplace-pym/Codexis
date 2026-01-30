function isBrokenPipeError(err) {
    const code = err?.code;
    return code === "EPIPE" || code === "EIO";
}
export function createSafeStreamWriter(options = {}) {
    let closed = false;
    let notified = false;
    const noteBrokenPipe = (err, stream) => {
        if (notified)
            return;
        notified = true;
        options.onBrokenPipe?.(err, stream);
    };
    const handleError = (err, stream) => {
        if (!isBrokenPipeError(err)) {
            throw err;
        }
        closed = true;
        noteBrokenPipe(err, stream);
        return false;
    };
    const write = (stream, text) => {
        if (closed)
            return false;
        try {
            options.beforeWrite?.();
        }
        catch (err) {
            return handleError(err, process.stderr);
        }
        try {
            stream.write(text);
            return !closed;
        }
        catch (err) {
            return handleError(err, stream);
        }
    };
    const writeLine = (stream, text) => write(stream, `${text}\n`);
    return {
        write,
        writeLine,
        reset: () => {
            closed = false;
            notified = false;
        },
        isClosed: () => closed,
    };
}
