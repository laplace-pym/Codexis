import { defaultRuntime } from "../runtime.js";
import { runConfigureWizard } from "./configure.wizard.js";
export async function configureCommand(runtime = defaultRuntime) {
    await runConfigureWizard({ command: "configure" }, runtime);
}
export async function configureCommandWithSections(sections, runtime = defaultRuntime) {
    await runConfigureWizard({ command: "configure", sections }, runtime);
}
