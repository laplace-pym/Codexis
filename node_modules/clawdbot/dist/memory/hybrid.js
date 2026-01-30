export function buildFtsQuery(raw) {
    const tokens = raw
        .match(/[A-Za-z0-9_]+/g)
        ?.map((t) => t.trim())
        .filter(Boolean) ?? [];
    if (tokens.length === 0)
        return null;
    const quoted = tokens.map((t) => `"${t.replaceAll('"', "")}"`);
    return quoted.join(" AND ");
}
export function bm25RankToScore(rank) {
    const normalized = Number.isFinite(rank) ? Math.max(0, rank) : 999;
    return 1 / (1 + normalized);
}
export function mergeHybridResults(params) {
    const byId = new Map();
    for (const r of params.vector) {
        byId.set(r.id, {
            id: r.id,
            path: r.path,
            startLine: r.startLine,
            endLine: r.endLine,
            source: r.source,
            snippet: r.snippet,
            vectorScore: r.vectorScore,
            textScore: 0,
        });
    }
    for (const r of params.keyword) {
        const existing = byId.get(r.id);
        if (existing) {
            existing.textScore = r.textScore;
            if (r.snippet && r.snippet.length > 0)
                existing.snippet = r.snippet;
        }
        else {
            byId.set(r.id, {
                id: r.id,
                path: r.path,
                startLine: r.startLine,
                endLine: r.endLine,
                source: r.source,
                snippet: r.snippet,
                vectorScore: 0,
                textScore: r.textScore,
            });
        }
    }
    const merged = Array.from(byId.values()).map((entry) => {
        const score = params.vectorWeight * entry.vectorScore + params.textWeight * entry.textScore;
        return {
            path: entry.path,
            startLine: entry.startLine,
            endLine: entry.endLine,
            score,
            snippet: entry.snippet,
            source: entry.source,
        };
    });
    return merged.sort((a, b) => b.score - a.score);
}
