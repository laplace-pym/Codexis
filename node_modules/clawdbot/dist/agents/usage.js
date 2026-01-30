const asFiniteNumber = (value) => {
    if (typeof value !== "number")
        return undefined;
    if (!Number.isFinite(value))
        return undefined;
    return value;
};
export function hasNonzeroUsage(usage) {
    if (!usage)
        return false;
    return [usage.input, usage.output, usage.cacheRead, usage.cacheWrite, usage.total].some((v) => typeof v === "number" && Number.isFinite(v) && v > 0);
}
export function normalizeUsage(raw) {
    if (!raw)
        return undefined;
    const input = asFiniteNumber(raw.input ?? raw.inputTokens ?? raw.input_tokens ?? raw.promptTokens ?? raw.prompt_tokens);
    const output = asFiniteNumber(raw.output ??
        raw.outputTokens ??
        raw.output_tokens ??
        raw.completionTokens ??
        raw.completion_tokens);
    const cacheRead = asFiniteNumber(raw.cacheRead ?? raw.cache_read ?? raw.cache_read_input_tokens);
    const cacheWrite = asFiniteNumber(raw.cacheWrite ?? raw.cache_write ?? raw.cache_creation_input_tokens);
    const total = asFiniteNumber(raw.total ?? raw.totalTokens ?? raw.total_tokens);
    if (input === undefined &&
        output === undefined &&
        cacheRead === undefined &&
        cacheWrite === undefined &&
        total === undefined) {
        return undefined;
    }
    return {
        input,
        output,
        cacheRead,
        cacheWrite,
        total,
    };
}
export function derivePromptTokens(usage) {
    if (!usage)
        return undefined;
    const input = usage.input ?? 0;
    const cacheRead = usage.cacheRead ?? 0;
    const cacheWrite = usage.cacheWrite ?? 0;
    const sum = input + cacheRead + cacheWrite;
    return sum > 0 ? sum : undefined;
}
