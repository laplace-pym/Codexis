import { randomUUID } from "node:crypto";
import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
function asRecord(value) {
    return typeof value === "object" && value !== null ? value : {};
}
function asString(value) {
    return typeof value === "string" ? value : undefined;
}
function asNumber(value) {
    return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}
function asBoolean(value) {
    return typeof value === "boolean" ? value : undefined;
}
export function parseCameraSnapPayload(value) {
    const obj = asRecord(value);
    const format = asString(obj.format);
    const base64 = asString(obj.base64);
    const width = asNumber(obj.width);
    const height = asNumber(obj.height);
    if (!format || !base64 || width === undefined || height === undefined) {
        throw new Error("invalid camera.snap payload");
    }
    return { format, base64, width, height };
}
export function parseCameraClipPayload(value) {
    const obj = asRecord(value);
    const format = asString(obj.format);
    const base64 = asString(obj.base64);
    const durationMs = asNumber(obj.durationMs);
    const hasAudio = asBoolean(obj.hasAudio);
    if (!format || !base64 || durationMs === undefined || hasAudio === undefined) {
        throw new Error("invalid camera.clip payload");
    }
    return { format, base64, durationMs, hasAudio };
}
export function cameraTempPath(opts) {
    const tmpDir = opts.tmpDir ?? os.tmpdir();
    const id = opts.id ?? randomUUID();
    const facingPart = opts.facing ? `-${opts.facing}` : "";
    const ext = opts.ext.startsWith(".") ? opts.ext : `.${opts.ext}`;
    return path.join(tmpDir, `clawdbot-camera-${opts.kind}${facingPart}-${id}${ext}`);
}
export async function writeBase64ToFile(filePath, base64) {
    const buf = Buffer.from(base64, "base64");
    await fs.writeFile(filePath, buf);
    return { path: filePath, bytes: buf.length };
}
