export function recordPluginInstall(cfg, update) {
    const { pluginId, ...record } = update;
    const installs = {
        ...cfg.plugins?.installs,
        [pluginId]: {
            ...cfg.plugins?.installs?.[pluginId],
            ...record,
            installedAt: record.installedAt ?? new Date().toISOString(),
        },
    };
    return {
        ...cfg,
        plugins: {
            ...cfg.plugins,
            installs: {
                ...installs,
                [pluginId]: installs[pluginId],
            },
        },
    };
}
