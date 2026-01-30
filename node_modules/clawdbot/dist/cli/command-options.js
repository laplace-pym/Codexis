export function hasExplicitOptions(command, names) {
    if (typeof command.getOptionValueSource !== "function") {
        return false;
    }
    return names.some((name) => command.getOptionValueSource(name) === "cli");
}
