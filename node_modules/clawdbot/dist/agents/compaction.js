import { estimateTokens, generateSummary } from "@mariozechner/pi-coding-agent";
import { DEFAULT_CONTEXT_TOKENS } from "./defaults.js";
export const BASE_CHUNK_RATIO = 0.4;
export const MIN_CHUNK_RATIO = 0.15;
export const SAFETY_MARGIN = 1.2; // 20% buffer for estimateTokens() inaccuracy
const DEFAULT_SUMMARY_FALLBACK = "No prior history.";
const DEFAULT_PARTS = 2;
const MERGE_SUMMARIES_INSTRUCTIONS = "Merge these partial summaries into a single cohesive summary. Preserve decisions," +
    " TODOs, open questions, and any constraints.";
export function estimateMessagesTokens(messages) {
    return messages.reduce((sum, message) => sum + estimateTokens(message), 0);
}
function normalizeParts(parts, messageCount) {
    if (!Number.isFinite(parts) || parts <= 1)
        return 1;
    return Math.min(Math.max(1, Math.floor(parts)), Math.max(1, messageCount));
}
export function splitMessagesByTokenShare(messages, parts = DEFAULT_PARTS) {
    if (messages.length === 0)
        return [];
    const normalizedParts = normalizeParts(parts, messages.length);
    if (normalizedParts <= 1)
        return [messages];
    const totalTokens = estimateMessagesTokens(messages);
    const targetTokens = totalTokens / normalizedParts;
    const chunks = [];
    let current = [];
    let currentTokens = 0;
    for (const message of messages) {
        const messageTokens = estimateTokens(message);
        if (chunks.length < normalizedParts - 1 &&
            current.length > 0 &&
            currentTokens + messageTokens > targetTokens) {
            chunks.push(current);
            current = [];
            currentTokens = 0;
        }
        current.push(message);
        currentTokens += messageTokens;
    }
    if (current.length > 0) {
        chunks.push(current);
    }
    return chunks;
}
export function chunkMessagesByMaxTokens(messages, maxTokens) {
    if (messages.length === 0)
        return [];
    const chunks = [];
    let currentChunk = [];
    let currentTokens = 0;
    for (const message of messages) {
        const messageTokens = estimateTokens(message);
        if (currentChunk.length > 0 && currentTokens + messageTokens > maxTokens) {
            chunks.push(currentChunk);
            currentChunk = [];
            currentTokens = 0;
        }
        currentChunk.push(message);
        currentTokens += messageTokens;
        if (messageTokens > maxTokens) {
            // Split oversized messages to avoid unbounded chunk growth.
            chunks.push(currentChunk);
            currentChunk = [];
            currentTokens = 0;
        }
    }
    if (currentChunk.length > 0) {
        chunks.push(currentChunk);
    }
    return chunks;
}
/**
 * Compute adaptive chunk ratio based on average message size.
 * When messages are large, we use smaller chunks to avoid exceeding model limits.
 */
export function computeAdaptiveChunkRatio(messages, contextWindow) {
    if (messages.length === 0)
        return BASE_CHUNK_RATIO;
    const totalTokens = estimateMessagesTokens(messages);
    const avgTokens = totalTokens / messages.length;
    // Apply safety margin to account for estimation inaccuracy
    const safeAvgTokens = avgTokens * SAFETY_MARGIN;
    const avgRatio = safeAvgTokens / contextWindow;
    // If average message is > 10% of context, reduce chunk ratio
    if (avgRatio > 0.1) {
        const reduction = Math.min(avgRatio * 2, BASE_CHUNK_RATIO - MIN_CHUNK_RATIO);
        return Math.max(MIN_CHUNK_RATIO, BASE_CHUNK_RATIO - reduction);
    }
    return BASE_CHUNK_RATIO;
}
/**
 * Check if a single message is too large to summarize.
 * If single message > 50% of context, it can't be summarized safely.
 */
export function isOversizedForSummary(msg, contextWindow) {
    const tokens = estimateTokens(msg) * SAFETY_MARGIN;
    return tokens > contextWindow * 0.5;
}
async function summarizeChunks(params) {
    if (params.messages.length === 0) {
        return params.previousSummary ?? DEFAULT_SUMMARY_FALLBACK;
    }
    const chunks = chunkMessagesByMaxTokens(params.messages, params.maxChunkTokens);
    let summary = params.previousSummary;
    for (const chunk of chunks) {
        summary = await generateSummary(chunk, params.model, params.reserveTokens, params.apiKey, params.signal, params.customInstructions, summary);
    }
    return summary ?? DEFAULT_SUMMARY_FALLBACK;
}
/**
 * Summarize with progressive fallback for handling oversized messages.
 * If full summarization fails, tries partial summarization excluding oversized messages.
 */
export async function summarizeWithFallback(params) {
    const { messages, contextWindow } = params;
    if (messages.length === 0) {
        return params.previousSummary ?? DEFAULT_SUMMARY_FALLBACK;
    }
    // Try full summarization first
    try {
        return await summarizeChunks(params);
    }
    catch (fullError) {
        console.warn(`Full summarization failed, trying partial: ${fullError instanceof Error ? fullError.message : String(fullError)}`);
    }
    // Fallback 1: Summarize only small messages, note oversized ones
    const smallMessages = [];
    const oversizedNotes = [];
    for (const msg of messages) {
        if (isOversizedForSummary(msg, contextWindow)) {
            const role = msg.role ?? "message";
            const tokens = estimateTokens(msg);
            oversizedNotes.push(`[Large ${role} (~${Math.round(tokens / 1000)}K tokens) omitted from summary]`);
        }
        else {
            smallMessages.push(msg);
        }
    }
    if (smallMessages.length > 0) {
        try {
            const partialSummary = await summarizeChunks({
                ...params,
                messages: smallMessages,
            });
            const notes = oversizedNotes.length > 0 ? `\n\n${oversizedNotes.join("\n")}` : "";
            return partialSummary + notes;
        }
        catch (partialError) {
            console.warn(`Partial summarization also failed: ${partialError instanceof Error ? partialError.message : String(partialError)}`);
        }
    }
    // Final fallback: Just note what was there
    return (`Context contained ${messages.length} messages (${oversizedNotes.length} oversized). ` +
        `Summary unavailable due to size limits.`);
}
export async function summarizeInStages(params) {
    const { messages } = params;
    if (messages.length === 0) {
        return params.previousSummary ?? DEFAULT_SUMMARY_FALLBACK;
    }
    const minMessagesForSplit = Math.max(2, params.minMessagesForSplit ?? 4);
    const parts = normalizeParts(params.parts ?? DEFAULT_PARTS, messages.length);
    const totalTokens = estimateMessagesTokens(messages);
    if (parts <= 1 || messages.length < minMessagesForSplit || totalTokens <= params.maxChunkTokens) {
        return summarizeWithFallback(params);
    }
    const splits = splitMessagesByTokenShare(messages, parts).filter((chunk) => chunk.length > 0);
    if (splits.length <= 1) {
        return summarizeWithFallback(params);
    }
    const partialSummaries = [];
    for (const chunk of splits) {
        partialSummaries.push(await summarizeWithFallback({
            ...params,
            messages: chunk,
            previousSummary: undefined,
        }));
    }
    if (partialSummaries.length === 1) {
        return partialSummaries[0];
    }
    const summaryMessages = partialSummaries.map((summary) => ({
        role: "user",
        content: summary,
        timestamp: Date.now(),
    }));
    const mergeInstructions = params.customInstructions
        ? `${MERGE_SUMMARIES_INSTRUCTIONS}\n\nAdditional focus:\n${params.customInstructions}`
        : MERGE_SUMMARIES_INSTRUCTIONS;
    return summarizeWithFallback({
        ...params,
        messages: summaryMessages,
        customInstructions: mergeInstructions,
    });
}
export function pruneHistoryForContextShare(params) {
    const maxHistoryShare = params.maxHistoryShare ?? 0.5;
    const budgetTokens = Math.max(1, Math.floor(params.maxContextTokens * maxHistoryShare));
    let keptMessages = params.messages;
    let droppedChunks = 0;
    let droppedMessages = 0;
    let droppedTokens = 0;
    const parts = normalizeParts(params.parts ?? DEFAULT_PARTS, keptMessages.length);
    while (keptMessages.length > 0 && estimateMessagesTokens(keptMessages) > budgetTokens) {
        const chunks = splitMessagesByTokenShare(keptMessages, parts);
        if (chunks.length <= 1)
            break;
        const [dropped, ...rest] = chunks;
        droppedChunks += 1;
        droppedMessages += dropped.length;
        droppedTokens += estimateMessagesTokens(dropped);
        keptMessages = rest.flat();
    }
    return {
        messages: keptMessages,
        droppedChunks,
        droppedMessages,
        droppedTokens,
        keptTokens: estimateMessagesTokens(keptMessages),
        budgetTokens,
    };
}
export function resolveContextWindowTokens(model) {
    return Math.max(1, Math.floor(model?.contextWindow ?? DEFAULT_CONTEXT_TOKENS));
}
