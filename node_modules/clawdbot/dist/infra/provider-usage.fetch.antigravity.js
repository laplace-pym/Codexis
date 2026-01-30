import { logDebug } from "../logger.js";
import { fetchJson } from "./provider-usage.fetch.shared.js";
import { clampPercent, PROVIDER_LABELS } from "./provider-usage.shared.js";
const BASE_URL = "https://cloudcode-pa.googleapis.com";
const LOAD_CODE_ASSIST_PATH = "/v1internal:loadCodeAssist";
const FETCH_AVAILABLE_MODELS_PATH = "/v1internal:fetchAvailableModels";
const METADATA = {
    ideType: "ANTIGRAVITY",
    platform: "PLATFORM_UNSPECIFIED",
    pluginType: "GEMINI",
};
function parseNumber(value) {
    if (typeof value === "number" && Number.isFinite(value))
        return value;
    if (typeof value === "string") {
        const parsed = Number.parseFloat(value);
        if (Number.isFinite(parsed))
            return parsed;
    }
    return undefined;
}
function parseEpochMs(isoString) {
    if (!isoString?.trim())
        return undefined;
    try {
        const ms = Date.parse(isoString);
        if (Number.isFinite(ms))
            return ms;
    }
    catch {
        // ignore parse errors
    }
    return undefined;
}
async function parseErrorMessage(res) {
    try {
        const data = (await res.json());
        const message = data?.error?.message?.trim();
        if (message)
            return message;
    }
    catch {
        // ignore parse errors
    }
    return `HTTP ${res.status}`;
}
function extractCredits(data) {
    const available = parseNumber(data.availablePromptCredits);
    const monthly = parseNumber(data.planInfo?.monthlyPromptCredits);
    if (available === undefined || monthly === undefined || monthly <= 0)
        return undefined;
    return { available, monthly };
}
function extractPlanInfo(data) {
    const tierName = data.currentTier?.name?.trim();
    if (tierName)
        return tierName;
    const planType = data.planType?.trim();
    if (planType)
        return planType;
    return undefined;
}
function extractProjectId(data) {
    const project = data.cloudaicompanionProject;
    if (!project)
        return undefined;
    if (typeof project === "string")
        return project.trim() ? project : undefined;
    const projectId = typeof project.id === "string" ? project.id.trim() : undefined;
    return projectId || undefined;
}
function extractModelQuotas(data) {
    const result = new Map();
    if (!data.models || typeof data.models !== "object")
        return result;
    for (const [modelId, modelInfo] of Object.entries(data.models)) {
        const quotaInfo = modelInfo.quotaInfo;
        if (!quotaInfo)
            continue;
        const remainingFraction = parseNumber(quotaInfo.remainingFraction);
        if (remainingFraction === undefined)
            continue;
        const resetTime = parseEpochMs(quotaInfo.resetTime);
        result.set(modelId, { remainingFraction, resetTime });
    }
    return result;
}
function buildUsageWindows(opts) {
    const windows = [];
    // Credits window (overall)
    if (opts.credits) {
        const { available, monthly } = opts.credits;
        const used = monthly - available;
        const usedPercent = clampPercent((used / monthly) * 100);
        windows.push({ label: "Credits", usedPercent });
    }
    // Individual model windows
    if (opts.modelQuotas && opts.modelQuotas.size > 0) {
        const modelWindows = [];
        for (const [modelId, quota] of opts.modelQuotas) {
            const lowerModelId = modelId.toLowerCase();
            // Skip internal models
            if (lowerModelId.includes("chat_") || lowerModelId.includes("tab_")) {
                continue;
            }
            const usedPercent = clampPercent((1 - quota.remainingFraction) * 100);
            const window = { label: modelId, usedPercent };
            if (quota.resetTime)
                window.resetAt = quota.resetTime;
            modelWindows.push(window);
        }
        // Sort by usage (highest first) and take top 10
        modelWindows.sort((a, b) => b.usedPercent - a.usedPercent);
        const topModels = modelWindows.slice(0, 10);
        logDebug(`[antigravity] Built ${topModels.length} model windows from ${opts.modelQuotas.size} total models`);
        for (const w of topModels) {
            logDebug(`[antigravity]   ${w.label}: ${w.usedPercent.toFixed(1)}% used${w.resetAt ? ` (resets at ${new Date(w.resetAt).toISOString()})` : ""}`);
        }
        windows.push(...topModels);
    }
    return windows;
}
export async function fetchAntigravityUsage(token, timeoutMs, fetchFn) {
    const headers = {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
        "User-Agent": "antigravity",
        "X-Goog-Api-Client": "google-cloud-sdk vscode_cloudshelleditor/0.1",
    };
    let credits;
    let modelQuotas;
    let planInfo;
    let lastError;
    let projectId;
    // Fetch loadCodeAssist (credits + plan info)
    try {
        const res = await fetchJson(`${BASE_URL}${LOAD_CODE_ASSIST_PATH}`, { method: "POST", headers, body: JSON.stringify({ metadata: METADATA }) }, timeoutMs, fetchFn);
        if (res.ok) {
            const data = (await res.json());
            // Extract project ID for subsequent calls
            projectId = extractProjectId(data);
            credits = extractCredits(data);
            planInfo = extractPlanInfo(data);
            logDebug(`[antigravity] Credits: ${credits ? `${credits.available}/${credits.monthly}` : "none"}${planInfo ? ` (plan: ${planInfo})` : ""}`);
        }
        else {
            lastError = await parseErrorMessage(res);
            // Fatal auth errors - stop early
            if (res.status === 401) {
                return {
                    provider: "google-antigravity",
                    displayName: PROVIDER_LABELS["google-antigravity"],
                    windows: [],
                    error: "Token expired",
                };
            }
        }
    }
    catch {
        lastError = "Network error";
    }
    // Fetch fetchAvailableModels (model quotas)
    if (!projectId) {
        logDebug("[antigravity] Missing project id; requesting available models without project");
    }
    try {
        const body = JSON.stringify(projectId ? { project: projectId } : {});
        const res = await fetchJson(`${BASE_URL}${FETCH_AVAILABLE_MODELS_PATH}`, { method: "POST", headers, body }, timeoutMs, fetchFn);
        if (res.ok) {
            const data = (await res.json());
            modelQuotas = extractModelQuotas(data);
            logDebug(`[antigravity] Extracted ${modelQuotas.size} model quotas from API`);
            for (const [modelId, quota] of modelQuotas) {
                logDebug(`[antigravity]   ${modelId}: ${(quota.remainingFraction * 100).toFixed(1)}% remaining${quota.resetTime ? ` (resets ${new Date(quota.resetTime).toISOString()})` : ""}`);
            }
        }
        else {
            const err = await parseErrorMessage(res);
            if (res.status === 401) {
                lastError = "Token expired";
            }
            else if (!lastError) {
                lastError = err;
            }
        }
    }
    catch {
        if (!lastError)
            lastError = "Network error";
    }
    // Build windows from available data
    const windows = buildUsageWindows({ credits, modelQuotas });
    // Return error only if we got nothing
    if (windows.length === 0 && lastError) {
        logDebug(`[antigravity] Returning error snapshot: ${lastError}`);
        return {
            provider: "google-antigravity",
            displayName: PROVIDER_LABELS["google-antigravity"],
            windows: [],
            error: lastError,
        };
    }
    const snapshot = {
        provider: "google-antigravity",
        displayName: PROVIDER_LABELS["google-antigravity"],
        windows,
        plan: planInfo,
    };
    logDebug(`[antigravity] Returning snapshot with ${windows.length} windows${planInfo ? ` (plan: ${planInfo})` : ""}`);
    logDebug(`[antigravity] Snapshot: ${JSON.stringify(snapshot, null, 2)}`);
    return snapshot;
}
