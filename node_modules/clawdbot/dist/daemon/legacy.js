import { findLegacyLaunchAgents, uninstallLegacyLaunchAgents } from "./launchd.js";
import { findLegacySystemdUnits, uninstallLegacySystemdUnits } from "./systemd.js";
function formatLegacyLaunchAgents(agents) {
    return agents.map((agent) => ({
        platform: "darwin",
        label: agent.label,
        detail: [
            agent.loaded ? "loaded" : "not loaded",
            agent.exists ? `plist: ${agent.plistPath}` : "plist missing",
        ].join(", "),
    }));
}
function formatLegacySystemdUnits(units) {
    return units.map((unit) => ({
        platform: "linux",
        label: `${unit.name}.service`,
        detail: [
            unit.enabled ? "enabled" : "disabled",
            unit.exists ? `unit: ${unit.unitPath}` : "unit missing",
        ].join(", "),
    }));
}
export async function findLegacyGatewayServices(env) {
    if (process.platform === "darwin") {
        const agents = await findLegacyLaunchAgents(env);
        return formatLegacyLaunchAgents(agents);
    }
    if (process.platform === "linux") {
        const units = await findLegacySystemdUnits(env);
        return formatLegacySystemdUnits(units);
    }
    return [];
}
export async function uninstallLegacyGatewayServices({ env, stdout, }) {
    if (process.platform === "darwin") {
        const agents = await uninstallLegacyLaunchAgents({ env, stdout });
        return formatLegacyLaunchAgents(agents);
    }
    if (process.platform === "linux") {
        const units = await uninstallLegacySystemdUnits({ env, stdout });
        return formatLegacySystemdUnits(units);
    }
    return [];
}
