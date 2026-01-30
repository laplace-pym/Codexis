import { VERSION } from "../../version.js";
import { resolveCliChannelOptions } from "../channel-options.js";
export function createProgramContext() {
    const channelOptions = resolveCliChannelOptions();
    return {
        programVersion: VERSION,
        channelOptions,
        messageChannelOptions: channelOptions.join("|"),
        agentChannelOptions: ["last", ...channelOptions].join("|"),
    };
}
