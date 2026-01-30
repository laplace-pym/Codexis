import { addChannelAllowFromStoreEntry, approveChannelPairingCode, listChannelPairingRequests, readChannelAllowFromStore, upsertChannelPairingRequest, } from "../pairing/pairing-store.js";
const PROVIDER = "telegram";
export async function readTelegramAllowFromStore(env = process.env) {
    return readChannelAllowFromStore(PROVIDER, env);
}
export async function addTelegramAllowFromStoreEntry(params) {
    return addChannelAllowFromStoreEntry({
        channel: PROVIDER,
        entry: params.entry,
        env: params.env,
    });
}
export async function listTelegramPairingRequests(env = process.env) {
    const list = await listChannelPairingRequests(PROVIDER, env);
    return list.map((r) => ({
        chatId: r.id,
        code: r.code,
        createdAt: r.createdAt,
        lastSeenAt: r.lastSeenAt,
        username: r.meta?.username,
        firstName: r.meta?.firstName,
        lastName: r.meta?.lastName,
    }));
}
export async function upsertTelegramPairingRequest(params) {
    return upsertChannelPairingRequest({
        channel: PROVIDER,
        id: String(params.chatId),
        env: params.env,
        meta: {
            username: params.username,
            firstName: params.firstName,
            lastName: params.lastName,
        },
    });
}
export async function approveTelegramPairingCode(params) {
    const res = await approveChannelPairingCode({
        channel: PROVIDER,
        code: params.code,
        env: params.env,
    });
    if (!res)
        return null;
    const entry = res.entry
        ? {
            chatId: res.entry.id,
            code: res.entry.code,
            createdAt: res.entry.createdAt,
            lastSeenAt: res.entry.lastSeenAt,
            username: res.entry.meta?.username,
            firstName: res.entry.meta?.firstName,
            lastName: res.entry.meta?.lastName,
        }
        : undefined;
    return { chatId: res.id, entry };
}
export async function resolveTelegramEffectiveAllowFrom(params) {
    const env = params.env ?? process.env;
    const cfgAllowFrom = (params.cfg.channels?.telegram?.allowFrom ?? [])
        .map((v) => String(v).trim())
        .filter(Boolean)
        .map((v) => v.replace(/^(telegram|tg):/i, ""))
        .filter((v) => v !== "*");
    const cfgGroupAllowFrom = (params.cfg.channels?.telegram?.groupAllowFrom ?? [])
        .map((v) => String(v).trim())
        .filter(Boolean)
        .map((v) => v.replace(/^(telegram|tg):/i, ""))
        .filter((v) => v !== "*");
    const storeAllowFrom = await readTelegramAllowFromStore(env);
    const dm = Array.from(new Set([...cfgAllowFrom, ...storeAllowFrom]));
    const group = Array.from(new Set([
        ...(cfgGroupAllowFrom.length > 0 ? cfgGroupAllowFrom : cfgAllowFrom),
        ...storeAllowFrom,
    ]));
    return { dm, group };
}
