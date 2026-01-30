import { registerSlackChannelEvents } from "./events/channels.js";
import { registerSlackMemberEvents } from "./events/members.js";
import { registerSlackMessageEvents } from "./events/messages.js";
import { registerSlackPinEvents } from "./events/pins.js";
import { registerSlackReactionEvents } from "./events/reactions.js";
export function registerSlackMonitorEvents(params) {
    registerSlackMessageEvents({
        ctx: params.ctx,
        handleSlackMessage: params.handleSlackMessage,
    });
    registerSlackReactionEvents({ ctx: params.ctx });
    registerSlackMemberEvents({ ctx: params.ctx });
    registerSlackChannelEvents({ ctx: params.ctx });
    registerSlackPinEvents({ ctx: params.ctx });
}
