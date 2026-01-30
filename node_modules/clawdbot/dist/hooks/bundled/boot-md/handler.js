import { createDefaultDeps } from "../../../cli/deps.js";
import { runBootOnce } from "../../../gateway/boot.js";
const runBootChecklist = async (event) => {
    if (event.type !== "gateway" || event.action !== "startup") {
        return;
    }
    const context = (event.context ?? {});
    if (!context.cfg || !context.workspaceDir) {
        return;
    }
    const deps = context.deps ?? createDefaultDeps();
    await runBootOnce({ cfg: context.cfg, deps, workspaceDir: context.workspaceDir });
};
export default runBootChecklist;
