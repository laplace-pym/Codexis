import { randomUUID } from "node:crypto";
import * as os from "node:os";
import * as path from "node:path";
function asRecord(value) {
    return typeof value === "object" && value !== null ? value : {};
}
function asString(value) {
    return typeof value === "string" ? value : undefined;
}
export function parseCanvasSnapshotPayload(value) {
    const obj = asRecord(value);
    const format = asString(obj.format);
    const base64 = asString(obj.base64);
    if (!format || !base64) {
        throw new Error("invalid canvas.snapshot payload");
    }
    return { format, base64 };
}
export function canvasSnapshotTempPath(opts) {
    const tmpDir = opts.tmpDir ?? os.tmpdir();
    const id = opts.id ?? randomUUID();
    const ext = opts.ext.startsWith(".") ? opts.ext : `.${opts.ext}`;
    return path.join(tmpDir, `clawdbot-canvas-snapshot-${id}${ext}`);
}
