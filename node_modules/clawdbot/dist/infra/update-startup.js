import fs from "node:fs/promises";
import path from "node:path";
import { resolveStateDir } from "../config/paths.js";
import { resolveClawdbotPackageRoot } from "./clawdbot-root.js";
import { compareSemverStrings, resolveNpmChannelTag, checkUpdateStatus } from "./update-check.js";
import { normalizeUpdateChannel, DEFAULT_PACKAGE_CHANNEL } from "./update-channels.js";
import { VERSION } from "../version.js";
import { formatCliCommand } from "../cli/command-format.js";
const UPDATE_CHECK_FILENAME = "update-check.json";
const UPDATE_CHECK_INTERVAL_MS = 24 * 60 * 60 * 1000;
function shouldSkipCheck(allowInTests) {
    if (allowInTests)
        return false;
    if (process.env.VITEST || process.env.NODE_ENV === "test")
        return true;
    return false;
}
async function readState(statePath) {
    try {
        const raw = await fs.readFile(statePath, "utf-8");
        const parsed = JSON.parse(raw);
        return parsed && typeof parsed === "object" ? parsed : {};
    }
    catch {
        return {};
    }
}
async function writeState(statePath, state) {
    await fs.mkdir(path.dirname(statePath), { recursive: true });
    await fs.writeFile(statePath, JSON.stringify(state, null, 2), "utf-8");
}
export async function runGatewayUpdateCheck(params) {
    if (shouldSkipCheck(Boolean(params.allowInTests)))
        return;
    if (params.isNixMode)
        return;
    if (params.cfg.update?.checkOnStart === false)
        return;
    const statePath = path.join(resolveStateDir(), UPDATE_CHECK_FILENAME);
    const state = await readState(statePath);
    const now = Date.now();
    const lastCheckedAt = state.lastCheckedAt ? Date.parse(state.lastCheckedAt) : null;
    if (lastCheckedAt && Number.isFinite(lastCheckedAt)) {
        if (now - lastCheckedAt < UPDATE_CHECK_INTERVAL_MS)
            return;
    }
    const root = await resolveClawdbotPackageRoot({
        moduleUrl: import.meta.url,
        argv1: process.argv[1],
        cwd: process.cwd(),
    });
    const status = await checkUpdateStatus({
        root,
        timeoutMs: 2500,
        fetchGit: false,
        includeRegistry: false,
    });
    const nextState = {
        ...state,
        lastCheckedAt: new Date(now).toISOString(),
    };
    if (status.installKind !== "package") {
        await writeState(statePath, nextState);
        return;
    }
    const channel = normalizeUpdateChannel(params.cfg.update?.channel) ?? DEFAULT_PACKAGE_CHANNEL;
    const resolved = await resolveNpmChannelTag({ channel, timeoutMs: 2500 });
    const tag = resolved.tag;
    if (!resolved.version) {
        await writeState(statePath, nextState);
        return;
    }
    const cmp = compareSemverStrings(VERSION, resolved.version);
    if (cmp != null && cmp < 0) {
        const shouldNotify = state.lastNotifiedVersion !== resolved.version || state.lastNotifiedTag !== tag;
        if (shouldNotify) {
            params.log.info(`update available (${tag}): v${resolved.version} (current v${VERSION}). Run: ${formatCliCommand("clawdbot update")}`);
            nextState.lastNotifiedVersion = resolved.version;
            nextState.lastNotifiedTag = tag;
        }
    }
    await writeState(statePath, nextState);
}
export function scheduleGatewayUpdateCheck(params) {
    void runGatewayUpdateCheck(params).catch(() => { });
}
