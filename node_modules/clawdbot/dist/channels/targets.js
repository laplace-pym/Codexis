export function normalizeTargetId(kind, id) {
    return `${kind}:${id}`.toLowerCase();
}
export function buildMessagingTarget(kind, id, raw) {
    return {
        kind,
        id,
        raw,
        normalized: normalizeTargetId(kind, id),
    };
}
export function ensureTargetId(params) {
    if (!params.pattern.test(params.candidate)) {
        throw new Error(params.errorMessage);
    }
    return params.candidate;
}
export function requireTargetKind(params) {
    const kindLabel = params.kind;
    if (!params.target) {
        throw new Error(`${params.platform} ${kindLabel} id is required.`);
    }
    if (params.target.kind !== params.kind) {
        throw new Error(`${params.platform} ${kindLabel} id is required (use ${kindLabel}:<id>).`);
    }
    return params.target.id;
}
