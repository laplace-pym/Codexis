export function mergeConfigSection(base, patch, options = {}) {
    const next = { ...(base ?? undefined) };
    for (const [key, value] of Object.entries(patch)) {
        if (value === undefined) {
            if (options.unsetOnUndefined?.includes(key)) {
                delete next[key];
            }
            continue;
        }
        next[key] = value;
    }
    return next;
}
export function mergeWhatsAppConfig(cfg, patch, options) {
    return {
        ...cfg,
        channels: {
            ...cfg.channels,
            whatsapp: mergeConfigSection(cfg.channels?.whatsapp, patch, options),
        },
    };
}
