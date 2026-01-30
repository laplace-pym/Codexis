import { ErrorCodes, errorShape, formatValidationErrors } from "../protocol/index.js";
import { formatForLog } from "../ws-log.js";
export function respondInvalidParams(params) {
    params.respond(false, undefined, errorShape(ErrorCodes.INVALID_REQUEST, `invalid ${params.method} params: ${formatValidationErrors(params.validator.errors)}`));
}
export async function respondUnavailableOnThrow(respond, fn) {
    try {
        await fn();
    }
    catch (err) {
        respond(false, undefined, errorShape(ErrorCodes.UNAVAILABLE, formatForLog(err)));
    }
}
export function uniqueSortedStrings(values) {
    return [...new Set(values.filter((v) => typeof v === "string"))]
        .map((v) => v.trim())
        .filter(Boolean)
        .sort();
}
export function safeParseJson(value) {
    if (typeof value !== "string")
        return undefined;
    const trimmed = value.trim();
    if (!trimmed)
        return undefined;
    try {
        return JSON.parse(trimmed);
    }
    catch {
        return { payloadJSON: value };
    }
}
