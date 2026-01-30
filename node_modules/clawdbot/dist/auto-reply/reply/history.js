import { CURRENT_MESSAGE_MARKER } from "./mentions.js";
export const HISTORY_CONTEXT_MARKER = "[Chat messages since your last reply - for context]";
export const DEFAULT_GROUP_HISTORY_LIMIT = 50;
export function buildHistoryContext(params) {
    const { historyText, currentMessage } = params;
    const lineBreak = params.lineBreak ?? "\n";
    if (!historyText.trim())
        return currentMessage;
    return [HISTORY_CONTEXT_MARKER, historyText, "", CURRENT_MESSAGE_MARKER, currentMessage].join(lineBreak);
}
export function appendHistoryEntry(params) {
    const { historyMap, historyKey, entry } = params;
    if (params.limit <= 0)
        return [];
    const history = historyMap.get(historyKey) ?? [];
    history.push(entry);
    while (history.length > params.limit)
        history.shift();
    historyMap.set(historyKey, history);
    return history;
}
export function recordPendingHistoryEntry(params) {
    return appendHistoryEntry(params);
}
export function recordPendingHistoryEntryIfEnabled(params) {
    if (!params.entry)
        return [];
    if (params.limit <= 0)
        return [];
    return recordPendingHistoryEntry({
        historyMap: params.historyMap,
        historyKey: params.historyKey,
        entry: params.entry,
        limit: params.limit,
    });
}
export function buildPendingHistoryContextFromMap(params) {
    if (params.limit <= 0)
        return params.currentMessage;
    const entries = params.historyMap.get(params.historyKey) ?? [];
    return buildHistoryContextFromEntries({
        entries,
        currentMessage: params.currentMessage,
        formatEntry: params.formatEntry,
        lineBreak: params.lineBreak,
        excludeLast: false,
    });
}
export function buildHistoryContextFromMap(params) {
    if (params.limit <= 0)
        return params.currentMessage;
    const entries = params.entry
        ? appendHistoryEntry({
            historyMap: params.historyMap,
            historyKey: params.historyKey,
            entry: params.entry,
            limit: params.limit,
        })
        : (params.historyMap.get(params.historyKey) ?? []);
    return buildHistoryContextFromEntries({
        entries,
        currentMessage: params.currentMessage,
        formatEntry: params.formatEntry,
        lineBreak: params.lineBreak,
        excludeLast: params.excludeLast,
    });
}
export function clearHistoryEntries(params) {
    params.historyMap.set(params.historyKey, []);
}
export function clearHistoryEntriesIfEnabled(params) {
    if (params.limit <= 0)
        return;
    clearHistoryEntries({ historyMap: params.historyMap, historyKey: params.historyKey });
}
export function buildHistoryContextFromEntries(params) {
    const lineBreak = params.lineBreak ?? "\n";
    const entries = params.excludeLast === false ? params.entries : params.entries.slice(0, -1);
    if (entries.length === 0)
        return params.currentMessage;
    const historyText = entries.map(params.formatEntry).join(lineBreak);
    return buildHistoryContext({
        historyText,
        currentMessage: params.currentMessage,
        lineBreak,
    });
}
