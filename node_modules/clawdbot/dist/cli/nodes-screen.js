import { randomUUID } from "node:crypto";
import * as os from "node:os";
import * as path from "node:path";
import { writeBase64ToFile } from "./nodes-camera.js";
function asRecord(value) {
    return typeof value === "object" && value !== null ? value : {};
}
function asString(value) {
    return typeof value === "string" ? value : undefined;
}
export function parseScreenRecordPayload(value) {
    const obj = asRecord(value);
    const format = asString(obj.format);
    const base64 = asString(obj.base64);
    if (!format || !base64) {
        throw new Error("invalid screen.record payload");
    }
    return {
        format,
        base64,
        durationMs: typeof obj.durationMs === "number" ? obj.durationMs : undefined,
        fps: typeof obj.fps === "number" ? obj.fps : undefined,
        screenIndex: typeof obj.screenIndex === "number" ? obj.screenIndex : undefined,
        hasAudio: typeof obj.hasAudio === "boolean" ? obj.hasAudio : undefined,
    };
}
export function screenRecordTempPath(opts) {
    const tmpDir = opts.tmpDir ?? os.tmpdir();
    const id = opts.id ?? randomUUID();
    const ext = opts.ext.startsWith(".") ? opts.ext : `.${opts.ext}`;
    return path.join(tmpDir, `clawdbot-screen-record-${id}${ext}`);
}
export async function writeScreenRecordToFile(filePath, base64) {
    return writeBase64ToFile(filePath, base64);
}
