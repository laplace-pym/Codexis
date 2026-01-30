import { resolveUserPath } from "../../../utils.js";
export function resolveNonInteractiveWorkspaceDir(params) {
    const raw = (params.opts.workspace ??
        params.baseConfig.agents?.defaults?.workspace ??
        params.defaultWorkspaceDir).trim();
    return resolveUserPath(raw);
}
