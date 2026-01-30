const isMobilePlatform = (platform) => {
    const p = typeof platform === "string" ? platform.trim().toLowerCase() : "";
    if (!p)
        return false;
    return p.startsWith("ios") || p.startsWith("ipados") || p.startsWith("android");
};
export function hasConnectedMobileNode(registry) {
    const connected = registry.listConnected();
    return connected.some((n) => isMobilePlatform(n.platform));
}
