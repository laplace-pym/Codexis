import { resolveAgentMaxConcurrent, resolveSubagentMaxConcurrent } from "../config/agent-limits.js";
import { setCommandLaneConcurrency } from "../process/command-queue.js";
export function applyGatewayLaneConcurrency(cfg) {
    setCommandLaneConcurrency("cron" /* CommandLane.Cron */, cfg.cron?.maxConcurrentRuns ?? 1);
    setCommandLaneConcurrency("main" /* CommandLane.Main */, resolveAgentMaxConcurrent(cfg));
    setCommandLaneConcurrency("subagent" /* CommandLane.Subagent */, resolveSubagentMaxConcurrent(cfg));
}
