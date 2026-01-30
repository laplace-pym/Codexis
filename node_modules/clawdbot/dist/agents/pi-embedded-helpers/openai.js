function parseOpenAIReasoningSignature(value) {
    if (!value)
        return null;
    let candidate = null;
    if (typeof value === "string") {
        const trimmed = value.trim();
        if (!trimmed.startsWith("{") || !trimmed.endsWith("}"))
            return null;
        try {
            candidate = JSON.parse(trimmed);
        }
        catch {
            return null;
        }
    }
    else if (typeof value === "object") {
        candidate = value;
    }
    if (!candidate)
        return null;
    const id = typeof candidate.id === "string" ? candidate.id : "";
    const type = typeof candidate.type === "string" ? candidate.type : "";
    if (!id.startsWith("rs_"))
        return null;
    if (type === "reasoning" || type.startsWith("reasoning.")) {
        return { id, type };
    }
    return null;
}
function hasFollowingNonThinkingBlock(content, index) {
    for (let i = index + 1; i < content.length; i++) {
        const block = content[i];
        if (!block || typeof block !== "object")
            return true;
        if (block.type !== "thinking")
            return true;
    }
    return false;
}
/**
 * OpenAI Responses API can reject transcripts that contain a standalone `reasoning` item id
 * without the required following item.
 *
 * Clawdbot persists provider-specific reasoning metadata in `thinkingSignature`; if that metadata
 * is incomplete, drop the block to keep history usable.
 */
export function downgradeOpenAIReasoningBlocks(messages) {
    const out = [];
    for (const msg of messages) {
        if (!msg || typeof msg !== "object") {
            out.push(msg);
            continue;
        }
        const role = msg.role;
        if (role !== "assistant") {
            out.push(msg);
            continue;
        }
        const assistantMsg = msg;
        if (!Array.isArray(assistantMsg.content)) {
            out.push(msg);
            continue;
        }
        let changed = false;
        const nextContent = [];
        for (let i = 0; i < assistantMsg.content.length; i++) {
            const block = assistantMsg.content[i];
            if (!block || typeof block !== "object") {
                nextContent.push(block);
                continue;
            }
            const record = block;
            if (record.type !== "thinking") {
                nextContent.push(block);
                continue;
            }
            const signature = parseOpenAIReasoningSignature(record.thinkingSignature);
            if (!signature) {
                nextContent.push(block);
                continue;
            }
            if (hasFollowingNonThinkingBlock(assistantMsg.content, i)) {
                nextContent.push(block);
                continue;
            }
            changed = true;
        }
        if (!changed) {
            out.push(msg);
            continue;
        }
        if (nextContent.length === 0) {
            continue;
        }
        out.push({ ...assistantMsg, content: nextContent });
    }
    return out;
}
