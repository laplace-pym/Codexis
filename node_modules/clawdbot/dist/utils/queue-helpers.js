export function elideQueueText(text, limit = 140) {
    if (text.length <= limit)
        return text;
    return `${text.slice(0, Math.max(0, limit - 1)).trimEnd()}…`;
}
export function buildQueueSummaryLine(text, limit = 160) {
    const cleaned = text.replace(/\s+/g, " ").trim();
    return elideQueueText(cleaned, limit);
}
export function shouldSkipQueueItem(params) {
    if (!params.dedupe)
        return false;
    return params.dedupe(params.item, params.items);
}
export function applyQueueDropPolicy(params) {
    const cap = params.queue.cap;
    if (cap <= 0 || params.queue.items.length < cap)
        return true;
    if (params.queue.dropPolicy === "new")
        return false;
    const dropCount = params.queue.items.length - cap + 1;
    const dropped = params.queue.items.splice(0, dropCount);
    if (params.queue.dropPolicy === "summarize") {
        for (const item of dropped) {
            params.queue.droppedCount += 1;
            params.queue.summaryLines.push(buildQueueSummaryLine(params.summarize(item)));
        }
        const limit = Math.max(0, params.summaryLimit ?? cap);
        while (params.queue.summaryLines.length > limit)
            params.queue.summaryLines.shift();
    }
    return true;
}
export function waitForQueueDebounce(queue) {
    const debounceMs = Math.max(0, queue.debounceMs);
    if (debounceMs <= 0)
        return Promise.resolve();
    return new Promise((resolve) => {
        const check = () => {
            const since = Date.now() - queue.lastEnqueuedAt;
            if (since >= debounceMs) {
                resolve();
                return;
            }
            setTimeout(check, debounceMs - since);
        };
        check();
    });
}
export function buildQueueSummaryPrompt(params) {
    if (params.state.dropPolicy !== "summarize" || params.state.droppedCount <= 0) {
        return undefined;
    }
    const noun = params.noun;
    const title = params.title ??
        `[Queue overflow] Dropped ${params.state.droppedCount} ${noun}${params.state.droppedCount === 1 ? "" : "s"} due to cap.`;
    const lines = [title];
    if (params.state.summaryLines.length > 0) {
        lines.push("Summary:");
        for (const line of params.state.summaryLines) {
            lines.push(`- ${line}`);
        }
    }
    params.state.droppedCount = 0;
    params.state.summaryLines = [];
    return lines.join("\n");
}
export function buildCollectPrompt(params) {
    const blocks = [params.title];
    if (params.summary)
        blocks.push(params.summary);
    params.items.forEach((item, idx) => {
        blocks.push(params.renderItem(item, idx));
    });
    return blocks.join("\n\n");
}
export function hasCrossChannelItems(items, resolveKey) {
    const keys = new Set();
    let hasUnkeyed = false;
    for (const item of items) {
        const resolved = resolveKey(item);
        if (resolved.cross)
            return true;
        if (!resolved.key) {
            hasUnkeyed = true;
            continue;
        }
        keys.add(resolved.key);
    }
    if (keys.size === 0)
        return false;
    if (hasUnkeyed)
        return true;
    return keys.size > 1;
}
