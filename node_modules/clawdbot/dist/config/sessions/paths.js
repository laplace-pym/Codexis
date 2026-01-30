import os from "node:os";
import path from "node:path";
import { DEFAULT_AGENT_ID, normalizeAgentId } from "../../routing/session-key.js";
import { resolveStateDir } from "../paths.js";
function resolveAgentSessionsDir(agentId, env = process.env, homedir = os.homedir) {
    const root = resolveStateDir(env, homedir);
    const id = normalizeAgentId(agentId ?? DEFAULT_AGENT_ID);
    return path.join(root, "agents", id, "sessions");
}
export function resolveSessionTranscriptsDir(env = process.env, homedir = os.homedir) {
    return resolveAgentSessionsDir(DEFAULT_AGENT_ID, env, homedir);
}
export function resolveSessionTranscriptsDirForAgent(agentId, env = process.env, homedir = os.homedir) {
    return resolveAgentSessionsDir(agentId, env, homedir);
}
export function resolveDefaultSessionStorePath(agentId) {
    return path.join(resolveAgentSessionsDir(agentId), "sessions.json");
}
export function resolveSessionTranscriptPath(sessionId, agentId, topicId) {
    const safeTopicId = typeof topicId === "string"
        ? encodeURIComponent(topicId)
        : typeof topicId === "number"
            ? String(topicId)
            : undefined;
    const fileName = safeTopicId !== undefined ? `${sessionId}-topic-${safeTopicId}.jsonl` : `${sessionId}.jsonl`;
    return path.join(resolveAgentSessionsDir(agentId), fileName);
}
export function resolveSessionFilePath(sessionId, entry, opts) {
    const candidate = entry?.sessionFile?.trim();
    return candidate ? candidate : resolveSessionTranscriptPath(sessionId, opts?.agentId);
}
export function resolveStorePath(store, opts) {
    const agentId = normalizeAgentId(opts?.agentId ?? DEFAULT_AGENT_ID);
    if (!store)
        return resolveDefaultSessionStorePath(agentId);
    if (store.includes("{agentId}")) {
        const expanded = store.replaceAll("{agentId}", agentId);
        if (expanded.startsWith("~")) {
            return path.resolve(expanded.replace(/^~(?=$|[\\/])/, os.homedir()));
        }
        return path.resolve(expanded);
    }
    if (store.startsWith("~"))
        return path.resolve(store.replace(/^~(?=$|[\\/])/, os.homedir()));
    return path.resolve(store);
}
