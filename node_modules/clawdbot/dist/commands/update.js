import { defaultRuntime } from "../runtime.js";
import { runConfigureWizard } from "./configure.js";
export async function updateCommand(runtime = defaultRuntime) {
    await runConfigureWizard({
        command: "update",
        sections: [
            "workspace",
            "model",
            "gateway",
            "daemon",
            "providers",
            "skills",
            "health",
        ],
    }, runtime);
}
