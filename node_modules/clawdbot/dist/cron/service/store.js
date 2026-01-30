import { migrateLegacyCronPayload } from "../payload-migration.js";
import { loadCronStore, saveCronStore } from "../store.js";
import { inferLegacyName, normalizeOptionalText } from "./normalize.js";
const storeCache = new Map();
export async function ensureLoaded(state) {
    if (state.store)
        return;
    const cached = storeCache.get(state.deps.storePath);
    if (cached) {
        state.store = cached;
        return;
    }
    const loaded = await loadCronStore(state.deps.storePath);
    const jobs = (loaded.jobs ?? []);
    let mutated = false;
    for (const raw of jobs) {
        const nameRaw = raw.name;
        if (typeof nameRaw !== "string" || nameRaw.trim().length === 0) {
            raw.name = inferLegacyName({
                schedule: raw.schedule,
                payload: raw.payload,
            });
            mutated = true;
        }
        else {
            raw.name = nameRaw.trim();
        }
        const desc = normalizeOptionalText(raw.description);
        if (raw.description !== desc) {
            raw.description = desc;
            mutated = true;
        }
        const payload = raw.payload;
        if (payload && typeof payload === "object" && !Array.isArray(payload)) {
            if (migrateLegacyCronPayload(payload)) {
                mutated = true;
            }
        }
    }
    state.store = { version: 1, jobs: jobs };
    storeCache.set(state.deps.storePath, state.store);
    if (mutated)
        await persist(state);
}
export function warnIfDisabled(state, action) {
    if (state.deps.cronEnabled)
        return;
    if (state.warnedDisabled)
        return;
    state.warnedDisabled = true;
    state.deps.log.warn({ enabled: false, action, storePath: state.deps.storePath }, "cron: scheduler disabled; jobs will not run automatically");
}
export async function persist(state) {
    if (!state.store)
        return;
    await saveCronStore(state.deps.storePath, state.store);
}
