import { logWebSelfId, sendMessageWhatsApp } from "../channels/web/index.js";
import { sendMessageDiscord } from "../discord/send.js";
import { sendMessageIMessage } from "../imessage/send.js";
import { sendMessageSignal } from "../signal/send.js";
import { sendMessageSlack } from "../slack/send.js";
import { sendMessageTelegram } from "../telegram/send.js";
export function createDefaultDeps() {
    return {
        sendMessageWhatsApp,
        sendMessageTelegram,
        sendMessageDiscord,
        sendMessageSlack,
        sendMessageSignal,
        sendMessageIMessage,
    };
}
// Provider docking: extend this mapping when adding new outbound send deps.
export function createOutboundSendDeps(deps) {
    return {
        sendWhatsApp: deps.sendMessageWhatsApp,
        sendTelegram: deps.sendMessageTelegram,
        sendDiscord: deps.sendMessageDiscord,
        sendSlack: deps.sendMessageSlack,
        sendSignal: deps.sendMessageSignal,
        sendIMessage: deps.sendMessageIMessage,
    };
}
export { logWebSelfId };
