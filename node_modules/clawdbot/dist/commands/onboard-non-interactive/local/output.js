export function logNonInteractiveOnboardingJson(params) {
    if (!params.opts.json)
        return;
    params.runtime.log(JSON.stringify({
        mode: params.mode,
        workspace: params.workspaceDir,
        authChoice: params.authChoice,
        gateway: params.gateway,
        installDaemon: Boolean(params.installDaemon),
        daemonRuntime: params.daemonRuntime,
        skipSkills: Boolean(params.skipSkills),
        skipHealth: Boolean(params.skipHealth),
    }, null, 2));
}
