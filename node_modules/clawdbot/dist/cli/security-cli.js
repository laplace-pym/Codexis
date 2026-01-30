import { loadConfig } from "../config/config.js";
import { defaultRuntime } from "../runtime.js";
import { runSecurityAudit } from "../security/audit.js";
import { fixSecurityFootguns } from "../security/fix.js";
import { formatDocsLink } from "../terminal/links.js";
import { isRich, theme } from "../terminal/theme.js";
import { shortenHomeInString, shortenHomePath } from "../utils.js";
import { formatCliCommand } from "./command-format.js";
function formatSummary(summary) {
    const rich = isRich();
    const c = summary.critical;
    const w = summary.warn;
    const i = summary.info;
    const parts = [];
    parts.push(rich ? theme.error(`${c} critical`) : `${c} critical`);
    parts.push(rich ? theme.warn(`${w} warn`) : `${w} warn`);
    parts.push(rich ? theme.muted(`${i} info`) : `${i} info`);
    return parts.join(" · ");
}
export function registerSecurityCli(program) {
    const security = program
        .command("security")
        .description("Security tools (audit)")
        .addHelpText("after", () => `\n${theme.muted("Docs:")} ${formatDocsLink("/cli/security", "docs.clawd.bot/cli/security")}\n`);
    security
        .command("audit")
        .description("Audit config + local state for common security foot-guns")
        .option("--deep", "Attempt live Gateway probe (best-effort)", false)
        .option("--fix", "Apply safe fixes (tighten defaults + chmod state/config)", false)
        .option("--json", "Print JSON", false)
        .action(async (opts) => {
        const fixResult = opts.fix ? await fixSecurityFootguns().catch((_err) => null) : null;
        const cfg = loadConfig();
        const report = await runSecurityAudit({
            config: cfg,
            deep: Boolean(opts.deep),
            includeFilesystem: true,
            includeChannelSecurity: true,
        });
        if (opts.json) {
            defaultRuntime.log(JSON.stringify(fixResult ? { fix: fixResult, report } : report, null, 2));
            return;
        }
        const rich = isRich();
        const heading = (text) => (rich ? theme.heading(text) : text);
        const muted = (text) => (rich ? theme.muted(text) : text);
        const lines = [];
        lines.push(heading("Clawdbot security audit"));
        lines.push(muted(`Summary: ${formatSummary(report.summary)}`));
        lines.push(muted(`Run deeper: ${formatCliCommand("clawdbot security audit --deep")}`));
        if (opts.fix) {
            lines.push(muted(`Fix: ${formatCliCommand("clawdbot security audit --fix")}`));
            if (!fixResult) {
                lines.push(muted("Fixes: failed to apply (unexpected error)"));
            }
            else if (fixResult.errors.length === 0 &&
                fixResult.changes.length === 0 &&
                fixResult.actions.every((a) => a.ok === false)) {
                lines.push(muted("Fixes: no changes applied"));
            }
            else {
                lines.push("");
                lines.push(heading("FIX"));
                for (const change of fixResult.changes) {
                    lines.push(muted(`  ${shortenHomeInString(change)}`));
                }
                for (const action of fixResult.actions) {
                    const mode = action.mode.toString(8).padStart(3, "0");
                    if (action.ok)
                        lines.push(muted(`  chmod ${mode} ${shortenHomePath(action.path)}`));
                    else if (action.skipped)
                        lines.push(muted(`  skip chmod ${mode} ${shortenHomePath(action.path)} (${action.skipped})`));
                    else if (action.error)
                        lines.push(muted(`  chmod ${mode} ${shortenHomePath(action.path)} failed: ${action.error}`));
                }
                if (fixResult.errors.length > 0) {
                    for (const err of fixResult.errors) {
                        lines.push(muted(`  error: ${shortenHomeInString(err)}`));
                    }
                }
            }
        }
        const bySeverity = (sev) => report.findings.filter((f) => f.severity === sev);
        const render = (sev) => {
            const list = bySeverity(sev);
            if (list.length === 0)
                return;
            const label = sev === "critical"
                ? rich
                    ? theme.error("CRITICAL")
                    : "CRITICAL"
                : sev === "warn"
                    ? rich
                        ? theme.warn("WARN")
                        : "WARN"
                    : rich
                        ? theme.muted("INFO")
                        : "INFO";
            lines.push("");
            lines.push(heading(label));
            for (const f of list) {
                lines.push(`${theme.muted(f.checkId)} ${f.title}`);
                lines.push(`  ${f.detail}`);
                if (f.remediation?.trim())
                    lines.push(`  ${muted(`Fix: ${f.remediation.trim()}`)}`);
            }
        };
        render("critical");
        render("warn");
        render("info");
        defaultRuntime.log(lines.join("\n"));
    });
}
