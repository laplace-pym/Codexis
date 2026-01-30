import crypto from "node:crypto";
import { logVerbose, danger } from "../globals.js";
function validateSignature(body, signature, channelSecret) {
    const hash = crypto.createHmac("SHA256", channelSecret).update(body).digest("base64");
    return hash === signature;
}
function readRawBody(req) {
    const rawBody = req.rawBody ??
        (typeof req.body === "string" || Buffer.isBuffer(req.body) ? req.body : null);
    if (!rawBody)
        return null;
    return Buffer.isBuffer(rawBody) ? rawBody.toString("utf-8") : rawBody;
}
function parseWebhookBody(req, rawBody) {
    if (req.body && typeof req.body === "object" && !Buffer.isBuffer(req.body)) {
        return req.body;
    }
    try {
        return JSON.parse(rawBody);
    }
    catch {
        return null;
    }
}
export function createLineWebhookMiddleware(options) {
    const { channelSecret, onEvents, runtime } = options;
    return async (req, res, _next) => {
        try {
            const signature = req.headers["x-line-signature"];
            if (!signature || typeof signature !== "string") {
                res.status(400).json({ error: "Missing X-Line-Signature header" });
                return;
            }
            const rawBody = readRawBody(req);
            if (!rawBody) {
                res.status(400).json({ error: "Missing raw request body for signature verification" });
                return;
            }
            if (!validateSignature(rawBody, signature, channelSecret)) {
                logVerbose("line: webhook signature validation failed");
                res.status(401).json({ error: "Invalid signature" });
                return;
            }
            const body = parseWebhookBody(req, rawBody);
            if (!body) {
                res.status(400).json({ error: "Invalid webhook payload" });
                return;
            }
            // Respond immediately to avoid timeout
            res.status(200).json({ status: "ok" });
            // Process events asynchronously
            if (body.events && body.events.length > 0) {
                logVerbose(`line: received ${body.events.length} webhook events`);
                await onEvents(body).catch((err) => {
                    runtime?.error?.(danger(`line webhook handler failed: ${String(err)}`));
                });
            }
        }
        catch (err) {
            runtime?.error?.(danger(`line webhook error: ${String(err)}`));
            if (!res.headersSent) {
                res.status(500).json({ error: "Internal server error" });
            }
        }
    };
}
export function startLineWebhook(options) {
    const path = options.path ?? "/line/webhook";
    const middleware = createLineWebhookMiddleware({
        channelSecret: options.channelSecret,
        onEvents: options.onEvents,
        runtime: options.runtime,
    });
    return { path, handler: middleware };
}
