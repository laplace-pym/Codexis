export const HEARTBEAT_TOKEN = "HEARTBEAT_OK";
export const SILENT_REPLY_TOKEN = "NO_REPLY";
function escapeRegExp(value) {
    return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
export function isSilentReplyText(text, token = SILENT_REPLY_TOKEN) {
    if (!text)
        return false;
    const escaped = escapeRegExp(token);
    const prefix = new RegExp(`^\\s*${escaped}(?=$|\\W)`);
    if (prefix.test(text))
        return true;
    const suffix = new RegExp(`\\b${escaped}\\b\\W*$`);
    return suffix.test(text);
}
