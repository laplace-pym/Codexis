const DEFAULT_VISIBILITY = {
    showOk: false, // Silent by default
    showAlerts: true, // Show content messages
    useIndicator: true, // Emit indicator events
};
export function resolveHeartbeatVisibility(params) {
    const { cfg, channel, accountId } = params;
    // Layer 1: Global channel defaults
    const channelDefaults = cfg.channels?.defaults?.heartbeat;
    // Layer 2: Per-channel config (at channel root level)
    const channelCfg = cfg.channels?.[channel];
    const perChannel = channelCfg?.heartbeat;
    // Layer 3: Per-account config (most specific)
    const accountCfg = accountId ? channelCfg?.accounts?.[accountId] : undefined;
    const perAccount = accountCfg?.heartbeat;
    // Precedence: per-account > per-channel > channel-defaults > global defaults
    return {
        showOk: perAccount?.showOk ??
            perChannel?.showOk ??
            channelDefaults?.showOk ??
            DEFAULT_VISIBILITY.showOk,
        showAlerts: perAccount?.showAlerts ??
            perChannel?.showAlerts ??
            channelDefaults?.showAlerts ??
            DEFAULT_VISIBILITY.showAlerts,
        useIndicator: perAccount?.useIndicator ??
            perChannel?.useIndicator ??
            channelDefaults?.useIndicator ??
            DEFAULT_VISIBILITY.useIndicator,
    };
}
