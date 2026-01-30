export function buildEmbeddedSandboxInfo(sandbox, execElevated) {
    if (!sandbox?.enabled)
        return undefined;
    const elevatedAllowed = Boolean(execElevated?.enabled && execElevated.allowed);
    return {
        enabled: true,
        workspaceDir: sandbox.workspaceDir,
        workspaceAccess: sandbox.workspaceAccess,
        agentWorkspaceMount: sandbox.workspaceAccess === "ro" ? "/agent" : undefined,
        browserControlUrl: sandbox.browser?.controlUrl,
        browserNoVncUrl: sandbox.browser?.noVncUrl,
        hostBrowserAllowed: sandbox.browserAllowHostControl,
        allowedControlUrls: sandbox.browserAllowedControlUrls,
        allowedControlHosts: sandbox.browserAllowedControlHosts,
        allowedControlPorts: sandbox.browserAllowedControlPorts,
        ...(elevatedAllowed
            ? {
                elevated: {
                    allowed: true,
                    defaultLevel: execElevated?.defaultLevel ?? "off",
                },
            }
            : {}),
    };
}
