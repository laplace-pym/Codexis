import { STATE_DIR_CLAWDBOT } from "../config/paths.js";
import { createSubsystemLogger } from "../logging/subsystem.js";
const log = createSubsystemLogger("plugins");
export async function startPluginServices(params) {
    const running = [];
    for (const entry of params.registry.services) {
        const service = entry.service;
        try {
            await service.start({
                config: params.config,
                workspaceDir: params.workspaceDir,
                stateDir: STATE_DIR_CLAWDBOT,
                logger: {
                    info: (msg) => log.info(msg),
                    warn: (msg) => log.warn(msg),
                    error: (msg) => log.error(msg),
                    debug: (msg) => log.debug(msg),
                },
            });
            running.push({
                id: service.id,
                stop: service.stop
                    ? () => service.stop?.({
                        config: params.config,
                        workspaceDir: params.workspaceDir,
                        stateDir: STATE_DIR_CLAWDBOT,
                        logger: {
                            info: (msg) => log.info(msg),
                            warn: (msg) => log.warn(msg),
                            error: (msg) => log.error(msg),
                            debug: (msg) => log.debug(msg),
                        },
                    })
                    : undefined,
            });
        }
        catch (err) {
            log.error(`plugin service failed (${service.id}): ${String(err)}`);
        }
    }
    return {
        stop: async () => {
            for (const entry of running.reverse()) {
                if (!entry.stop)
                    continue;
                try {
                    await entry.stop();
                }
                catch (err) {
                    log.warn(`plugin service stop failed (${entry.id}): ${String(err)}`);
                }
            }
        },
    };
}
