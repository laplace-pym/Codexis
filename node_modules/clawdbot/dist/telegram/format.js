import { chunkMarkdownIR, markdownToIR, } from "../markdown/ir.js";
import { renderMarkdownWithMarkers } from "../markdown/render.js";
function escapeHtml(text) {
    return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
function escapeHtmlAttr(text) {
    return escapeHtml(text).replace(/"/g, "&quot;");
}
function buildTelegramLink(link, _text) {
    const href = link.href.trim();
    if (!href)
        return null;
    if (link.start === link.end)
        return null;
    const safeHref = escapeHtmlAttr(href);
    return {
        start: link.start,
        end: link.end,
        open: `<a href="${safeHref}">`,
        close: "</a>",
    };
}
function renderTelegramHtml(ir) {
    return renderMarkdownWithMarkers(ir, {
        styleMarkers: {
            bold: { open: "<b>", close: "</b>" },
            italic: { open: "<i>", close: "</i>" },
            strikethrough: { open: "<s>", close: "</s>" },
            code: { open: "<code>", close: "</code>" },
            code_block: { open: "<pre><code>", close: "</code></pre>" },
        },
        escapeText: escapeHtml,
        buildLink: buildTelegramLink,
    });
}
export function markdownToTelegramHtml(markdown, options = {}) {
    const ir = markdownToIR(markdown ?? "", {
        linkify: true,
        headingStyle: "none",
        blockquotePrefix: "",
        tableMode: options.tableMode,
    });
    return renderTelegramHtml(ir);
}
export function renderTelegramHtmlText(text, options = {}) {
    const textMode = options.textMode ?? "markdown";
    if (textMode === "html")
        return text;
    return markdownToTelegramHtml(text, { tableMode: options.tableMode });
}
export function markdownToTelegramChunks(markdown, limit, options = {}) {
    const ir = markdownToIR(markdown ?? "", {
        linkify: true,
        headingStyle: "none",
        blockquotePrefix: "",
        tableMode: options.tableMode,
    });
    const chunks = chunkMarkdownIR(ir, limit);
    return chunks.map((chunk) => ({
        html: renderTelegramHtml(chunk),
        text: chunk.text,
    }));
}
export function markdownToTelegramHtmlChunks(markdown, limit) {
    return markdownToTelegramChunks(markdown, limit).map((chunk) => chunk.html);
}
