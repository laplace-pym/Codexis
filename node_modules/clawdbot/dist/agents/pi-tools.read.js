import { createEditTool, createReadTool, createWriteTool } from "@mariozechner/pi-coding-agent";
import { detectMime } from "../media/mime.js";
import { assertSandboxPath } from "./sandbox-paths.js";
import { sanitizeToolResultImages } from "./tool-images.js";
async function sniffMimeFromBase64(base64) {
    const trimmed = base64.trim();
    if (!trimmed)
        return undefined;
    const take = Math.min(256, trimmed.length);
    const sliceLen = take - (take % 4);
    if (sliceLen < 8)
        return undefined;
    try {
        const head = Buffer.from(trimmed.slice(0, sliceLen), "base64");
        return await detectMime({ buffer: head });
    }
    catch {
        return undefined;
    }
}
function rewriteReadImageHeader(text, mimeType) {
    // pi-coding-agent uses: "Read image file [image/png]"
    if (text.startsWith("Read image file [") && text.endsWith("]")) {
        return `Read image file [${mimeType}]`;
    }
    return text;
}
async function normalizeReadImageResult(result, filePath) {
    const content = Array.isArray(result.content) ? result.content : [];
    const image = content.find((b) => !!b &&
        typeof b === "object" &&
        b.type === "image" &&
        typeof b.data === "string" &&
        typeof b.mimeType === "string");
    if (!image)
        return result;
    if (!image.data.trim()) {
        throw new Error(`read: image payload is empty (${filePath})`);
    }
    const sniffed = await sniffMimeFromBase64(image.data);
    if (!sniffed)
        return result;
    if (!sniffed.startsWith("image/")) {
        throw new Error(`read: file looks like ${sniffed} but was treated as ${image.mimeType} (${filePath})`);
    }
    if (sniffed === image.mimeType)
        return result;
    const nextContent = content.map((block) => {
        if (block && typeof block === "object" && block.type === "image") {
            const b = block;
            return { ...b, mimeType: sniffed };
        }
        if (block &&
            typeof block === "object" &&
            block.type === "text" &&
            typeof block.text === "string") {
            const b = block;
            return {
                ...b,
                text: rewriteReadImageHeader(b.text, sniffed),
            };
        }
        return block;
    });
    return { ...result, content: nextContent };
}
export const CLAUDE_PARAM_GROUPS = {
    read: [{ keys: ["path", "file_path"], label: "path (path or file_path)" }],
    write: [{ keys: ["path", "file_path"], label: "path (path or file_path)" }],
    edit: [
        { keys: ["path", "file_path"], label: "path (path or file_path)" },
        {
            keys: ["oldText", "old_string"],
            label: "oldText (oldText or old_string)",
        },
        {
            keys: ["newText", "new_string"],
            label: "newText (newText or new_string)",
        },
    ],
};
// Normalize tool parameters from Claude Code conventions to pi-coding-agent conventions.
// Claude Code uses file_path/old_string/new_string while pi-coding-agent uses path/oldText/newText.
// This prevents models trained on Claude Code from getting stuck in tool-call loops.
export function normalizeToolParams(params) {
    if (!params || typeof params !== "object")
        return undefined;
    const record = params;
    const normalized = { ...record };
    // file_path → path (read, write, edit)
    if ("file_path" in normalized && !("path" in normalized)) {
        normalized.path = normalized.file_path;
        delete normalized.file_path;
    }
    // old_string → oldText (edit)
    if ("old_string" in normalized && !("oldText" in normalized)) {
        normalized.oldText = normalized.old_string;
        delete normalized.old_string;
    }
    // new_string → newText (edit)
    if ("new_string" in normalized && !("newText" in normalized)) {
        normalized.newText = normalized.new_string;
        delete normalized.new_string;
    }
    return normalized;
}
export function patchToolSchemaForClaudeCompatibility(tool) {
    const schema = tool.parameters && typeof tool.parameters === "object"
        ? tool.parameters
        : undefined;
    if (!schema || !schema.properties || typeof schema.properties !== "object") {
        return tool;
    }
    const properties = { ...schema.properties };
    const required = Array.isArray(schema.required)
        ? schema.required.filter((key) => typeof key === "string")
        : [];
    let changed = false;
    const aliasPairs = [
        { original: "path", alias: "file_path" },
        { original: "oldText", alias: "old_string" },
        { original: "newText", alias: "new_string" },
    ];
    for (const { original, alias } of aliasPairs) {
        if (!(original in properties))
            continue;
        if (!(alias in properties)) {
            properties[alias] = properties[original];
            changed = true;
        }
        const idx = required.indexOf(original);
        if (idx !== -1) {
            required.splice(idx, 1);
            changed = true;
        }
    }
    if (!changed)
        return tool;
    return {
        ...tool,
        parameters: {
            ...schema,
            properties,
            ...(required.length > 0 ? { required } : {}),
        },
    };
}
export function assertRequiredParams(record, groups, toolName) {
    if (!record || typeof record !== "object") {
        throw new Error(`Missing parameters for ${toolName}`);
    }
    for (const group of groups) {
        const satisfied = group.keys.some((key) => {
            if (!(key in record))
                return false;
            const value = record[key];
            if (typeof value !== "string")
                return false;
            if (group.allowEmpty)
                return true;
            return value.trim().length > 0;
        });
        if (!satisfied) {
            const label = group.label ?? group.keys.join(" or ");
            throw new Error(`Missing required parameter: ${label}`);
        }
    }
}
// Generic wrapper to normalize parameters for any tool
export function wrapToolParamNormalization(tool, requiredParamGroups) {
    const patched = patchToolSchemaForClaudeCompatibility(tool);
    return {
        ...patched,
        execute: async (toolCallId, params, signal, onUpdate) => {
            const normalized = normalizeToolParams(params);
            const record = normalized ??
                (params && typeof params === "object" ? params : undefined);
            if (requiredParamGroups?.length) {
                assertRequiredParams(record, requiredParamGroups, tool.name);
            }
            return tool.execute(toolCallId, normalized ?? params, signal, onUpdate);
        },
    };
}
function wrapSandboxPathGuard(tool, root) {
    return {
        ...tool,
        execute: async (toolCallId, args, signal, onUpdate) => {
            const normalized = normalizeToolParams(args);
            const record = normalized ??
                (args && typeof args === "object" ? args : undefined);
            const filePath = record?.path;
            if (typeof filePath === "string" && filePath.trim()) {
                await assertSandboxPath({ filePath, cwd: root, root });
            }
            return tool.execute(toolCallId, normalized ?? args, signal, onUpdate);
        },
    };
}
export function createSandboxedReadTool(root) {
    const base = createReadTool(root);
    return wrapSandboxPathGuard(createClawdbotReadTool(base), root);
}
export function createSandboxedWriteTool(root) {
    const base = createWriteTool(root);
    return wrapSandboxPathGuard(wrapToolParamNormalization(base, CLAUDE_PARAM_GROUPS.write), root);
}
export function createSandboxedEditTool(root) {
    const base = createEditTool(root);
    return wrapSandboxPathGuard(wrapToolParamNormalization(base, CLAUDE_PARAM_GROUPS.edit), root);
}
export function createClawdbotReadTool(base) {
    const patched = patchToolSchemaForClaudeCompatibility(base);
    return {
        ...patched,
        execute: async (toolCallId, params, signal) => {
            const normalized = normalizeToolParams(params);
            const record = normalized ??
                (params && typeof params === "object" ? params : undefined);
            assertRequiredParams(record, CLAUDE_PARAM_GROUPS.read, base.name);
            const result = (await base.execute(toolCallId, normalized ?? params, signal));
            const filePath = typeof record?.path === "string" ? String(record.path) : "<unknown>";
            const normalizedResult = await normalizeReadImageResult(result, filePath);
            return sanitizeToolResultImages(normalizedResult, `read:${filePath}`);
        },
    };
}
