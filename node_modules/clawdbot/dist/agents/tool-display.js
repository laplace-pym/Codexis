import { redactToolDetail } from "../logging/redact.js";
import { shortenHomeInString } from "../utils.js";
import TOOL_DISPLAY_JSON from "./tool-display.json" with { type: "json" };
const TOOL_DISPLAY_CONFIG = TOOL_DISPLAY_JSON;
const FALLBACK = TOOL_DISPLAY_CONFIG.fallback ?? { emoji: "🧩" };
const TOOL_MAP = TOOL_DISPLAY_CONFIG.tools ?? {};
const DETAIL_LABEL_OVERRIDES = {
    agentId: "agent",
    sessionKey: "session",
    targetId: "target",
    targetUrl: "url",
    nodeId: "node",
    requestId: "request",
    messageId: "message",
    threadId: "thread",
    channelId: "channel",
    guildId: "guild",
    userId: "user",
    runTimeoutSeconds: "timeout",
    timeoutSeconds: "timeout",
    includeTools: "tools",
    pollQuestion: "poll",
    maxChars: "max chars",
};
const MAX_DETAIL_ENTRIES = 8;
function normalizeToolName(name) {
    return (name ?? "tool").trim();
}
function defaultTitle(name) {
    const cleaned = name.replace(/_/g, " ").trim();
    if (!cleaned)
        return "Tool";
    return cleaned
        .split(/\s+/)
        .map((part) => part.length <= 2 && part.toUpperCase() === part
        ? part
        : `${part.at(0)?.toUpperCase() ?? ""}${part.slice(1)}`)
        .join(" ");
}
function normalizeVerb(value) {
    const trimmed = value?.trim();
    if (!trimmed)
        return undefined;
    return trimmed.replace(/_/g, " ");
}
function coerceDisplayValue(value) {
    if (value === null || value === undefined)
        return undefined;
    if (typeof value === "string") {
        const trimmed = value.trim();
        if (!trimmed)
            return undefined;
        const firstLine = trimmed.split(/\r?\n/)[0]?.trim() ?? "";
        if (!firstLine)
            return undefined;
        return firstLine.length > 160 ? `${firstLine.slice(0, 157)}…` : firstLine;
    }
    if (typeof value === "boolean") {
        return value ? "true" : undefined;
    }
    if (typeof value === "number") {
        if (!Number.isFinite(value) || value === 0)
            return undefined;
        return String(value);
    }
    if (Array.isArray(value)) {
        const values = value
            .map((item) => coerceDisplayValue(item))
            .filter((item) => Boolean(item));
        if (values.length === 0)
            return undefined;
        const preview = values.slice(0, 3).join(", ");
        return values.length > 3 ? `${preview}…` : preview;
    }
    return undefined;
}
function lookupValueByPath(args, path) {
    if (!args || typeof args !== "object")
        return undefined;
    let current = args;
    for (const segment of path.split(".")) {
        if (!segment)
            return undefined;
        if (!current || typeof current !== "object")
            return undefined;
        const record = current;
        current = record[segment];
    }
    return current;
}
function formatDetailKey(raw) {
    const segments = raw.split(".").filter(Boolean);
    const last = segments.at(-1) ?? raw;
    const override = DETAIL_LABEL_OVERRIDES[last];
    if (override)
        return override;
    const cleaned = last.replace(/_/g, " ").replace(/-/g, " ");
    const spaced = cleaned.replace(/([a-z0-9])([A-Z])/g, "$1 $2");
    return spaced.trim().toLowerCase() || last.toLowerCase();
}
function resolveDetailFromKeys(args, keys) {
    const entries = [];
    for (const key of keys) {
        const value = lookupValueByPath(args, key);
        const display = coerceDisplayValue(value);
        if (!display)
            continue;
        entries.push({ label: formatDetailKey(key), value: display });
    }
    if (entries.length === 0)
        return undefined;
    if (entries.length === 1)
        return entries[0].value;
    const seen = new Set();
    const unique = [];
    for (const entry of entries) {
        const token = `${entry.label}:${entry.value}`;
        if (seen.has(token))
            continue;
        seen.add(token);
        unique.push(entry);
    }
    if (unique.length === 0)
        return undefined;
    return unique
        .slice(0, MAX_DETAIL_ENTRIES)
        .map((entry) => `${entry.label} ${entry.value}`)
        .join(" · ");
}
function resolveReadDetail(args) {
    if (!args || typeof args !== "object")
        return undefined;
    const record = args;
    const path = typeof record.path === "string" ? record.path : undefined;
    if (!path)
        return undefined;
    const offset = typeof record.offset === "number" ? record.offset : undefined;
    const limit = typeof record.limit === "number" ? record.limit : undefined;
    if (offset !== undefined && limit !== undefined) {
        return `${path}:${offset}-${offset + limit}`;
    }
    return path;
}
function resolveWriteDetail(args) {
    if (!args || typeof args !== "object")
        return undefined;
    const record = args;
    const path = typeof record.path === "string" ? record.path : undefined;
    return path;
}
function resolveActionSpec(spec, action) {
    if (!spec || !action)
        return undefined;
    return spec.actions?.[action] ?? undefined;
}
export function resolveToolDisplay(params) {
    const name = normalizeToolName(params.name);
    const key = name.toLowerCase();
    const spec = TOOL_MAP[key];
    const emoji = spec?.emoji ?? FALLBACK.emoji ?? "🧩";
    const title = spec?.title ?? defaultTitle(name);
    const label = spec?.label ?? title;
    const actionRaw = params.args && typeof params.args === "object"
        ? params.args.action
        : undefined;
    const action = typeof actionRaw === "string" ? actionRaw.trim() : undefined;
    const actionSpec = resolveActionSpec(spec, action);
    const verb = normalizeVerb(actionSpec?.label ?? action);
    let detail;
    if (key === "read")
        detail = resolveReadDetail(params.args);
    if (!detail && (key === "write" || key === "edit" || key === "attach")) {
        detail = resolveWriteDetail(params.args);
    }
    const detailKeys = actionSpec?.detailKeys ?? spec?.detailKeys ?? FALLBACK.detailKeys ?? [];
    if (!detail && detailKeys.length > 0) {
        detail = resolveDetailFromKeys(params.args, detailKeys);
    }
    if (!detail && params.meta) {
        detail = params.meta;
    }
    if (detail) {
        detail = shortenHomeInString(detail);
    }
    return {
        name,
        emoji,
        title,
        label,
        verb,
        detail,
    };
}
export function formatToolDetail(display) {
    const parts = [];
    if (display.verb)
        parts.push(display.verb);
    if (display.detail)
        parts.push(redactToolDetail(display.detail));
    if (parts.length === 0)
        return undefined;
    return parts.join(" · ");
}
export function formatToolSummary(display) {
    const detail = formatToolDetail(display);
    return detail
        ? `${display.emoji} ${display.label}: ${detail}`
        : `${display.emoji} ${display.label}`;
}
