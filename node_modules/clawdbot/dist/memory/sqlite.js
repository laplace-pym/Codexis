import { createRequire } from "node:module";
import { installProcessWarningFilter } from "../infra/warnings.js";
const require = createRequire(import.meta.url);
export function requireNodeSqlite() {
    installProcessWarningFilter();
    return require("node:sqlite");
}
