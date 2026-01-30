import { classifyFailoverReason } from "./pi-embedded-helpers.js";
const TIMEOUT_HINT_RE = /timeout|timed out|deadline exceeded|context deadline exceeded/i;
const ABORT_TIMEOUT_RE = /request was aborted|request aborted/i;
export class FailoverError extends Error {
    reason;
    provider;
    model;
    profileId;
    status;
    code;
    constructor(message, params) {
        super(message, { cause: params.cause });
        this.name = "FailoverError";
        this.reason = params.reason;
        this.provider = params.provider;
        this.model = params.model;
        this.profileId = params.profileId;
        this.status = params.status;
        this.code = params.code;
    }
}
export function isFailoverError(err) {
    return err instanceof FailoverError;
}
export function resolveFailoverStatus(reason) {
    switch (reason) {
        case "billing":
            return 402;
        case "rate_limit":
            return 429;
        case "auth":
            return 401;
        case "timeout":
            return 408;
        case "format":
            return 400;
        default:
            return undefined;
    }
}
function getStatusCode(err) {
    if (!err || typeof err !== "object")
        return undefined;
    const candidate = err.status ??
        err.statusCode;
    if (typeof candidate === "number")
        return candidate;
    if (typeof candidate === "string" && /^\d+$/.test(candidate)) {
        return Number(candidate);
    }
    return undefined;
}
function getErrorName(err) {
    if (!err || typeof err !== "object")
        return "";
    return "name" in err ? String(err.name) : "";
}
function getErrorCode(err) {
    if (!err || typeof err !== "object")
        return undefined;
    const candidate = err.code;
    if (typeof candidate !== "string")
        return undefined;
    const trimmed = candidate.trim();
    return trimmed ? trimmed : undefined;
}
function getErrorMessage(err) {
    if (err instanceof Error)
        return err.message;
    if (typeof err === "string")
        return err;
    if (typeof err === "number" || typeof err === "boolean" || typeof err === "bigint") {
        return String(err);
    }
    if (typeof err === "symbol")
        return err.description ?? "";
    if (err && typeof err === "object") {
        const message = err.message;
        if (typeof message === "string")
            return message;
    }
    return "";
}
function hasTimeoutHint(err) {
    if (!err)
        return false;
    if (getErrorName(err) === "TimeoutError")
        return true;
    const message = getErrorMessage(err);
    return Boolean(message && TIMEOUT_HINT_RE.test(message));
}
export function isTimeoutError(err) {
    if (hasTimeoutHint(err))
        return true;
    if (!err || typeof err !== "object")
        return false;
    if (getErrorName(err) !== "AbortError")
        return false;
    const message = getErrorMessage(err);
    if (message && ABORT_TIMEOUT_RE.test(message))
        return true;
    const cause = "cause" in err ? err.cause : undefined;
    const reason = "reason" in err ? err.reason : undefined;
    return hasTimeoutHint(cause) || hasTimeoutHint(reason);
}
export function resolveFailoverReasonFromError(err) {
    if (isFailoverError(err))
        return err.reason;
    const status = getStatusCode(err);
    if (status === 402)
        return "billing";
    if (status === 429)
        return "rate_limit";
    if (status === 401 || status === 403)
        return "auth";
    if (status === 408)
        return "timeout";
    const code = (getErrorCode(err) ?? "").toUpperCase();
    if (["ETIMEDOUT", "ESOCKETTIMEDOUT", "ECONNRESET", "ECONNABORTED"].includes(code)) {
        return "timeout";
    }
    if (isTimeoutError(err))
        return "timeout";
    const message = getErrorMessage(err);
    if (!message)
        return null;
    return classifyFailoverReason(message);
}
export function describeFailoverError(err) {
    if (isFailoverError(err)) {
        return {
            message: err.message,
            reason: err.reason,
            status: err.status,
            code: err.code,
        };
    }
    const message = getErrorMessage(err) || String(err);
    return {
        message,
        reason: resolveFailoverReasonFromError(err) ?? undefined,
        status: getStatusCode(err),
        code: getErrorCode(err),
    };
}
export function coerceToFailoverError(err, context) {
    if (isFailoverError(err))
        return err;
    const reason = resolveFailoverReasonFromError(err);
    if (!reason)
        return null;
    const message = getErrorMessage(err) || String(err);
    const status = getStatusCode(err) ?? resolveFailoverStatus(reason);
    const code = getErrorCode(err);
    return new FailoverError(message, {
        reason,
        provider: context?.provider,
        model: context?.model,
        profileId: context?.profileId,
        status,
        code,
        cause: err instanceof Error ? err : undefined,
    });
}
