import { getFileExtension } from "./mime.js";
const VOICE_AUDIO_EXTENSIONS = new Set([".oga", ".ogg", ".opus"]);
export function isVoiceCompatibleAudio(opts) {
    const mime = opts.contentType?.toLowerCase();
    if (mime && (mime.includes("ogg") || mime.includes("opus"))) {
        return true;
    }
    const fileName = opts.fileName?.trim();
    if (!fileName)
        return false;
    const ext = getFileExtension(fileName);
    if (!ext)
        return false;
    return VOICE_AUDIO_EXTENSIONS.has(ext);
}
