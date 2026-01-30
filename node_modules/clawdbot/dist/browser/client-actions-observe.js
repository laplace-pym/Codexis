import { fetchBrowserJson } from "./client-fetch.js";
function buildProfileQuery(profile) {
    return profile ? `?profile=${encodeURIComponent(profile)}` : "";
}
export async function browserConsoleMessages(baseUrl, opts = {}) {
    const q = new URLSearchParams();
    if (opts.level)
        q.set("level", opts.level);
    if (opts.targetId)
        q.set("targetId", opts.targetId);
    if (opts.profile)
        q.set("profile", opts.profile);
    const suffix = q.toString() ? `?${q.toString()}` : "";
    return await fetchBrowserJson(`${baseUrl}/console${suffix}`, { timeoutMs: 20000 });
}
export async function browserPdfSave(baseUrl, opts = {}) {
    const q = buildProfileQuery(opts.profile);
    return await fetchBrowserJson(`${baseUrl}/pdf${q}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ targetId: opts.targetId }),
        timeoutMs: 20000,
    });
}
export async function browserPageErrors(baseUrl, opts = {}) {
    const q = new URLSearchParams();
    if (opts.targetId)
        q.set("targetId", opts.targetId);
    if (typeof opts.clear === "boolean")
        q.set("clear", String(opts.clear));
    if (opts.profile)
        q.set("profile", opts.profile);
    const suffix = q.toString() ? `?${q.toString()}` : "";
    return await fetchBrowserJson(`${baseUrl}/errors${suffix}`, { timeoutMs: 20000 });
}
export async function browserRequests(baseUrl, opts = {}) {
    const q = new URLSearchParams();
    if (opts.targetId)
        q.set("targetId", opts.targetId);
    if (opts.filter)
        q.set("filter", opts.filter);
    if (typeof opts.clear === "boolean")
        q.set("clear", String(opts.clear));
    if (opts.profile)
        q.set("profile", opts.profile);
    const suffix = q.toString() ? `?${q.toString()}` : "";
    return await fetchBrowserJson(`${baseUrl}/requests${suffix}`, { timeoutMs: 20000 });
}
export async function browserTraceStart(baseUrl, opts = {}) {
    const q = buildProfileQuery(opts.profile);
    return await fetchBrowserJson(`${baseUrl}/trace/start${q}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            targetId: opts.targetId,
            screenshots: opts.screenshots,
            snapshots: opts.snapshots,
            sources: opts.sources,
        }),
        timeoutMs: 20000,
    });
}
export async function browserTraceStop(baseUrl, opts = {}) {
    const q = buildProfileQuery(opts.profile);
    return await fetchBrowserJson(`${baseUrl}/trace/stop${q}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ targetId: opts.targetId, path: opts.path }),
        timeoutMs: 20000,
    });
}
export async function browserHighlight(baseUrl, opts) {
    const q = buildProfileQuery(opts.profile);
    return await fetchBrowserJson(`${baseUrl}/highlight${q}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ targetId: opts.targetId, ref: opts.ref }),
        timeoutMs: 20000,
    });
}
export async function browserResponseBody(baseUrl, opts) {
    const q = buildProfileQuery(opts.profile);
    return await fetchBrowserJson(`${baseUrl}/response/body${q}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            targetId: opts.targetId,
            url: opts.url,
            timeoutMs: opts.timeoutMs,
            maxChars: opts.maxChars,
        }),
        timeoutMs: 20000,
    });
}
