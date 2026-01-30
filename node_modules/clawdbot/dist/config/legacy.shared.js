export const isRecord = (value) => Boolean(value && typeof value === "object" && !Array.isArray(value));
export const getRecord = (value) => isRecord(value) ? value : null;
export const ensureRecord = (root, key) => {
    const existing = root[key];
    if (isRecord(existing))
        return existing;
    const next = {};
    root[key] = next;
    return next;
};
export const mergeMissing = (target, source) => {
    for (const [key, value] of Object.entries(source)) {
        if (value === undefined)
            continue;
        const existing = target[key];
        if (existing === undefined) {
            target[key] = value;
            continue;
        }
        if (isRecord(existing) && isRecord(value)) {
            mergeMissing(existing, value);
        }
    }
};
const AUDIO_TRANSCRIPTION_CLI_ALLOWLIST = new Set(["whisper"]);
export const mapLegacyAudioTranscription = (value) => {
    const transcriber = getRecord(value);
    const command = Array.isArray(transcriber?.command) ? transcriber?.command : null;
    if (!command || command.length === 0)
        return null;
    const rawExecutable = String(command[0] ?? "").trim();
    if (!rawExecutable)
        return null;
    const executableName = rawExecutable.split(/[\\/]/).pop() ?? rawExecutable;
    if (!AUDIO_TRANSCRIPTION_CLI_ALLOWLIST.has(executableName))
        return null;
    const args = command.slice(1).map((part) => String(part));
    const timeoutSeconds = typeof transcriber?.timeoutSeconds === "number" ? transcriber?.timeoutSeconds : undefined;
    const result = { command: rawExecutable, type: "cli" };
    if (args.length > 0)
        result.args = args;
    if (timeoutSeconds !== undefined)
        result.timeoutSeconds = timeoutSeconds;
    return result;
};
export const getAgentsList = (agents) => {
    const list = agents?.list;
    return Array.isArray(list) ? list : [];
};
export const resolveDefaultAgentIdFromRaw = (raw) => {
    const agents = getRecord(raw.agents);
    const list = getAgentsList(agents);
    const defaultEntry = list.find((entry) => isRecord(entry) &&
        entry.default === true &&
        typeof entry.id === "string" &&
        entry.id.trim() !== "");
    if (defaultEntry)
        return defaultEntry.id.trim();
    const routing = getRecord(raw.routing);
    const routingDefault = typeof routing?.defaultAgentId === "string" ? routing.defaultAgentId.trim() : "";
    if (routingDefault)
        return routingDefault;
    const firstEntry = list.find((entry) => isRecord(entry) && typeof entry.id === "string" && entry.id.trim() !== "");
    if (firstEntry)
        return firstEntry.id.trim();
    return "main";
};
export const ensureAgentEntry = (list, id) => {
    const normalized = id.trim();
    const existing = list.find((entry) => isRecord(entry) && typeof entry.id === "string" && entry.id.trim() === normalized);
    if (existing)
        return existing;
    const created = { id: normalized };
    list.push(created);
    return created;
};
