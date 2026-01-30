const lineHttpRoutes = new Map();
export function normalizeLineWebhookPath(path) {
    const trimmed = path?.trim();
    if (!trimmed)
        return "/line/webhook";
    return trimmed.startsWith("/") ? trimmed : `/${trimmed}`;
}
export function registerLineHttpHandler(params) {
    const normalizedPath = normalizeLineWebhookPath(params.path);
    if (lineHttpRoutes.has(normalizedPath)) {
        const suffix = params.accountId ? ` for account "${params.accountId}"` : "";
        params.log?.(`line: webhook path ${normalizedPath} already registered${suffix}`);
        return () => { };
    }
    lineHttpRoutes.set(normalizedPath, params.handler);
    return () => {
        lineHttpRoutes.delete(normalizedPath);
    };
}
export async function handleLineHttpRequest(req, res) {
    const url = new URL(req.url ?? "/", "http://localhost");
    const handler = lineHttpRoutes.get(url.pathname);
    if (!handler)
        return false;
    await handler(req, res);
    return true;
}
