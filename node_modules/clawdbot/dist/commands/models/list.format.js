import { colorize, isRich as isRichTerminal, theme } from "../../terminal/theme.js";
export const isRich = (opts) => Boolean(isRichTerminal() && !opts?.json && !opts?.plain);
export const pad = (value, size) => value.padEnd(size);
export const formatKey = (key, rich) => colorize(rich, theme.warn, key);
export const formatValue = (value, rich) => colorize(rich, theme.info, value);
export const formatKeyValue = (key, value, rich, valueColor = theme.info) => `${formatKey(key, rich)}=${colorize(rich, valueColor, value)}`;
export const formatSeparator = (rich) => colorize(rich, theme.muted, " | ");
export const formatTag = (tag, rich) => {
    if (!rich)
        return tag;
    if (tag === "default")
        return theme.success(tag);
    if (tag === "image")
        return theme.accentBright(tag);
    if (tag === "configured")
        return theme.accent(tag);
    if (tag === "missing")
        return theme.error(tag);
    if (tag.startsWith("fallback#"))
        return theme.warn(tag);
    if (tag.startsWith("img-fallback#"))
        return theme.warn(tag);
    if (tag.startsWith("alias:"))
        return theme.accentDim(tag);
    return theme.muted(tag);
};
export const truncate = (value, max) => {
    if (value.length <= max)
        return value;
    if (max <= 3)
        return value.slice(0, max);
    return `${value.slice(0, max - 3)}...`;
};
export const maskApiKey = (value) => {
    const trimmed = value.trim();
    if (!trimmed)
        return "missing";
    if (trimmed.length <= 16)
        return trimmed;
    return `${trimmed.slice(0, 8)}...${trimmed.slice(-8)}`;
};
