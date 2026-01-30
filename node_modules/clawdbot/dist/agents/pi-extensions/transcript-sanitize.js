/**
 * Transcript repair/sanitization extension.
 *
 * Runs on every context build to prevent strict provider request rejections:
 * - duplicate or displaced tool results (Anthropic-compatible APIs, MiniMax, Cloud Code Assist)
 * - Cloud Code Assist tool call ID constraints + collision-safe sanitization
 */
import { isGoogleModelApi } from "../pi-embedded-helpers.js";
import { repairToolUseResultPairing } from "../session-transcript-repair.js";
import { sanitizeToolCallIdsForCloudCodeAssist } from "../tool-call-id.js";
export default function transcriptSanitizeExtension(api) {
    api.on("context", (event, ctx) => {
        let next = event.messages;
        const repaired = repairToolUseResultPairing(next);
        if (repaired.messages !== next)
            next = repaired.messages;
        if (isGoogleModelApi(ctx.model?.api)) {
            const repairedIds = sanitizeToolCallIdsForCloudCodeAssist(next);
            if (repairedIds !== next)
                next = repairedIds;
        }
        if (next === event.messages)
            return undefined;
        return { messages: next };
    });
}
