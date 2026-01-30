import { readChannelAllowFromStore } from "../../pairing/pairing-store.js";
import { allowListMatches, normalizeAllowList, normalizeAllowListLower } from "./allow-list.js";
export async function resolveSlackEffectiveAllowFrom(ctx) {
    const storeAllowFrom = await readChannelAllowFromStore("slack").catch(() => []);
    const allowFrom = normalizeAllowList([...ctx.allowFrom, ...storeAllowFrom]);
    const allowFromLower = normalizeAllowListLower(allowFrom);
    return { allowFrom, allowFromLower };
}
export function isSlackSenderAllowListed(params) {
    const { allowListLower, senderId, senderName } = params;
    return (allowListLower.length === 0 ||
        allowListMatches({
            allowList: allowListLower,
            id: senderId,
            name: senderName,
        }));
}
