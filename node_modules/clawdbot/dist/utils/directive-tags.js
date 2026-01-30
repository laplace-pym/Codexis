const AUDIO_TAG_RE = /\[\[\s*audio_as_voice\s*\]\]/gi;
const REPLY_TAG_RE = /\[\[\s*(?:reply_to_current|reply_to\s*:\s*([^\]\n]+))\s*\]\]/gi;
function normalizeDirectiveWhitespace(text) {
    return text
        .replace(/[ \t]+/g, " ")
        .replace(/[ \t]*\n[ \t]*/g, "\n")
        .trim();
}
export function parseInlineDirectives(text, options = {}) {
    const { currentMessageId, stripAudioTag = true, stripReplyTags = true } = options;
    if (!text) {
        return {
            text: "",
            audioAsVoice: false,
            replyToCurrent: false,
            hasAudioTag: false,
            hasReplyTag: false,
        };
    }
    let cleaned = text;
    let audioAsVoice = false;
    let hasAudioTag = false;
    let hasReplyTag = false;
    let sawCurrent = false;
    let lastExplicitId;
    cleaned = cleaned.replace(AUDIO_TAG_RE, (match) => {
        audioAsVoice = true;
        hasAudioTag = true;
        return stripAudioTag ? " " : match;
    });
    cleaned = cleaned.replace(REPLY_TAG_RE, (match, idRaw) => {
        hasReplyTag = true;
        if (idRaw === undefined) {
            sawCurrent = true;
        }
        else {
            const id = idRaw.trim();
            if (id)
                lastExplicitId = id;
        }
        return stripReplyTags ? " " : match;
    });
    cleaned = normalizeDirectiveWhitespace(cleaned);
    const replyToId = lastExplicitId ?? (sawCurrent ? currentMessageId?.trim() || undefined : undefined);
    return {
        text: cleaned,
        audioAsVoice,
        replyToId,
        replyToExplicitId: lastExplicitId,
        replyToCurrent: sawCurrent,
        hasAudioTag,
        hasReplyTag,
    };
}
