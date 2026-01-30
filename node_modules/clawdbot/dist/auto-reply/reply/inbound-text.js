export function normalizeInboundTextNewlines(input) {
    return input.replaceAll("\r\n", "\n").replaceAll("\r", "\n").replaceAll("\\n", "\n");
}
