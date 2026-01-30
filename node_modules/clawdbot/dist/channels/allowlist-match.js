export function formatAllowlistMatchMeta(match) {
    return `matchKey=${match?.matchKey ?? "none"} matchSource=${match?.matchSource ?? "none"}`;
}
