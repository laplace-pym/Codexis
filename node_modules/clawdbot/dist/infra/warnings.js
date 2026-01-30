const warningFilterKey = Symbol.for("clawdbot.warning-filter");
function shouldIgnoreWarning(warning) {
    if (warning.code === "DEP0040" && warning.message?.includes("punycode")) {
        return true;
    }
    if (warning.code === "DEP0060" && warning.message?.includes("util._extend")) {
        return true;
    }
    if (warning.name === "ExperimentalWarning" &&
        warning.message?.includes("SQLite is an experimental feature")) {
        return true;
    }
    return false;
}
export function installProcessWarningFilter() {
    const globalState = globalThis;
    if (globalState[warningFilterKey]?.installed)
        return;
    globalState[warningFilterKey] = { installed: true };
    process.on("warning", (warning) => {
        if (shouldIgnoreWarning(warning))
            return;
        process.stderr.write(`${warning.stack ?? warning.toString()}\n`);
    });
}
