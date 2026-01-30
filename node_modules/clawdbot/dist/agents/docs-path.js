import fs from "node:fs";
import path from "node:path";
import { resolveClawdbotPackageRoot } from "../infra/clawdbot-root.js";
export async function resolveClawdbotDocsPath(params) {
    const workspaceDir = params.workspaceDir?.trim();
    if (workspaceDir) {
        const workspaceDocs = path.join(workspaceDir, "docs");
        if (fs.existsSync(workspaceDocs))
            return workspaceDocs;
    }
    const packageRoot = await resolveClawdbotPackageRoot({
        cwd: params.cwd,
        argv1: params.argv1,
        moduleUrl: params.moduleUrl,
    });
    if (!packageRoot)
        return null;
    const packageDocs = path.join(packageRoot, "docs");
    return fs.existsSync(packageDocs) ? packageDocs : null;
}
