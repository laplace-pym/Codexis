import { loadConfig } from "../../config/config.js";
import { loadCostUsageSummary } from "../../infra/session-cost-usage.js";
import { loadProviderUsageSummary } from "../../infra/provider-usage.js";
const COST_USAGE_CACHE_TTL_MS = 30_000;
const costUsageCache = new Map();
const parseDays = (raw) => {
    if (typeof raw === "number" && Number.isFinite(raw))
        return Math.floor(raw);
    if (typeof raw === "string" && raw.trim() !== "") {
        const parsed = Number(raw);
        if (Number.isFinite(parsed))
            return Math.floor(parsed);
    }
    return 30;
};
async function loadCostUsageSummaryCached(params) {
    const days = Math.max(1, params.days);
    const now = Date.now();
    const cached = costUsageCache.get(days);
    if (cached?.summary && cached.updatedAt && now - cached.updatedAt < COST_USAGE_CACHE_TTL_MS) {
        return cached.summary;
    }
    if (cached?.inFlight) {
        if (cached.summary)
            return cached.summary;
        return await cached.inFlight;
    }
    const entry = cached ?? {};
    const inFlight = loadCostUsageSummary({ days, config: params.config })
        .then((summary) => {
        costUsageCache.set(days, { summary, updatedAt: Date.now() });
        return summary;
    })
        .catch((err) => {
        if (entry.summary)
            return entry.summary;
        throw err;
    })
        .finally(() => {
        const current = costUsageCache.get(days);
        if (current?.inFlight === inFlight) {
            current.inFlight = undefined;
            costUsageCache.set(days, current);
        }
    });
    entry.inFlight = inFlight;
    costUsageCache.set(days, entry);
    if (entry.summary)
        return entry.summary;
    return await inFlight;
}
export const usageHandlers = {
    "usage.status": async ({ respond }) => {
        const summary = await loadProviderUsageSummary();
        respond(true, summary, undefined);
    },
    "usage.cost": async ({ respond, params }) => {
        const config = loadConfig();
        const days = parseDays(params?.days);
        const summary = await loadCostUsageSummaryCached({ days, config });
        respond(true, summary, undefined);
    },
};
