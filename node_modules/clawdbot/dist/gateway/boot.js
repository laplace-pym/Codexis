import fs from "node:fs/promises";
import path from "node:path";
import { SILENT_REPLY_TOKEN } from "../auto-reply/tokens.js";
import { resolveMainSessionKey } from "../config/sessions/main-session.js";
import { agentCommand } from "../commands/agent.js";
import { createSubsystemLogger } from "../logging/subsystem.js";
import { defaultRuntime } from "../runtime.js";
const log = createSubsystemLogger("gateway/boot");
const BOOT_FILENAME = "BOOT.md";
function buildBootPrompt(content) {
    return [
        "You are running a boot check. Follow BOOT.md instructions exactly.",
        "",
        "BOOT.md:",
        content,
        "",
        "If BOOT.md asks you to send a message, use the message tool (action=send with channel + target).",
        "Use the `target` field (not `to`) for message tool destinations.",
        `After sending with the message tool, reply with ONLY: ${SILENT_REPLY_TOKEN}.`,
        `If nothing needs attention, reply with ONLY: ${SILENT_REPLY_TOKEN}.`,
    ].join("\n");
}
async function loadBootFile(workspaceDir) {
    const bootPath = path.join(workspaceDir, BOOT_FILENAME);
    try {
        const content = await fs.readFile(bootPath, "utf-8");
        const trimmed = content.trim();
        if (!trimmed)
            return { status: "empty" };
        return { status: "ok", content: trimmed };
    }
    catch (err) {
        const anyErr = err;
        if (anyErr.code === "ENOENT")
            return { status: "missing" };
        throw err;
    }
}
export async function runBootOnce(params) {
    const bootRuntime = {
        log: () => { },
        error: (message) => log.error(String(message)),
        exit: defaultRuntime.exit,
    };
    let result;
    try {
        result = await loadBootFile(params.workspaceDir);
    }
    catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        log.error(`boot: failed to read ${BOOT_FILENAME}: ${message}`);
        return { status: "failed", reason: message };
    }
    if (result.status === "missing" || result.status === "empty") {
        return { status: "skipped", reason: result.status };
    }
    const sessionKey = resolveMainSessionKey(params.cfg);
    const message = buildBootPrompt(result.content ?? "");
    try {
        await agentCommand({
            message,
            sessionKey,
            deliver: false,
        }, bootRuntime, params.deps);
        return { status: "ran" };
    }
    catch (err) {
        const messageText = err instanceof Error ? err.message : String(err);
        log.error(`boot: agent run failed: ${messageText}`);
        return { status: "failed", reason: messageText };
    }
}
