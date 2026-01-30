import { registerBrowserElementCommands } from "./register.element.js";
import { registerBrowserFilesAndDownloadsCommands } from "./register.files-downloads.js";
import { registerBrowserFormWaitEvalCommands } from "./register.form-wait-eval.js";
import { registerBrowserNavigationCommands } from "./register.navigation.js";
export function registerBrowserActionInputCommands(browser, parentOpts) {
    registerBrowserNavigationCommands(browser, parentOpts);
    registerBrowserElementCommands(browser, parentOpts);
    registerBrowserFilesAndDownloadsCommands(browser, parentOpts);
    registerBrowserFormWaitEvalCommands(browser, parentOpts);
}
