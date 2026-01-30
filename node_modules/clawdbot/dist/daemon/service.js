import { installLaunchAgent, isLaunchAgentLoaded, readLaunchAgentProgramArguments, readLaunchAgentRuntime, restartLaunchAgent, stopLaunchAgent, uninstallLaunchAgent, } from "./launchd.js";
import { installScheduledTask, isScheduledTaskInstalled, readScheduledTaskCommand, readScheduledTaskRuntime, restartScheduledTask, stopScheduledTask, uninstallScheduledTask, } from "./schtasks.js";
import { installSystemdService, isSystemdServiceEnabled, readSystemdServiceExecStart, readSystemdServiceRuntime, restartSystemdService, stopSystemdService, uninstallSystemdService, } from "./systemd.js";
export function resolveGatewayService() {
    if (process.platform === "darwin") {
        return {
            label: "LaunchAgent",
            loadedText: "loaded",
            notLoadedText: "not loaded",
            install: async (args) => {
                await installLaunchAgent(args);
            },
            uninstall: async (args) => {
                await uninstallLaunchAgent(args);
            },
            stop: async (args) => {
                await stopLaunchAgent({
                    stdout: args.stdout,
                    env: args.env,
                });
            },
            restart: async (args) => {
                await restartLaunchAgent({
                    stdout: args.stdout,
                    env: args.env,
                });
            },
            isLoaded: async (args) => isLaunchAgentLoaded(args),
            readCommand: readLaunchAgentProgramArguments,
            readRuntime: readLaunchAgentRuntime,
        };
    }
    if (process.platform === "linux") {
        return {
            label: "systemd",
            loadedText: "enabled",
            notLoadedText: "disabled",
            install: async (args) => {
                await installSystemdService(args);
            },
            uninstall: async (args) => {
                await uninstallSystemdService(args);
            },
            stop: async (args) => {
                await stopSystemdService({
                    stdout: args.stdout,
                    env: args.env,
                });
            },
            restart: async (args) => {
                await restartSystemdService({
                    stdout: args.stdout,
                    env: args.env,
                });
            },
            isLoaded: async (args) => isSystemdServiceEnabled(args),
            readCommand: readSystemdServiceExecStart,
            readRuntime: async (env) => await readSystemdServiceRuntime(env),
        };
    }
    if (process.platform === "win32") {
        return {
            label: "Scheduled Task",
            loadedText: "registered",
            notLoadedText: "missing",
            install: async (args) => {
                await installScheduledTask(args);
            },
            uninstall: async (args) => {
                await uninstallScheduledTask(args);
            },
            stop: async (args) => {
                await stopScheduledTask({
                    stdout: args.stdout,
                    env: args.env,
                });
            },
            restart: async (args) => {
                await restartScheduledTask({
                    stdout: args.stdout,
                    env: args.env,
                });
            },
            isLoaded: async (args) => isScheduledTaskInstalled(args),
            readCommand: readScheduledTaskCommand,
            readRuntime: async (env) => await readScheduledTaskRuntime(env),
        };
    }
    throw new Error(`Gateway service install not supported on ${process.platform}`);
}
