// Provider docking: add new providers here (order + meta + aliases), then
// register the plugin in src/providers/plugins/index.ts and keep protocol IDs in sync.
export const CHAT_PROVIDER_ORDER = [
    "telegram",
    "whatsapp",
    "discord",
    "slack",
    "signal",
    "imessage",
    "msteams",
];
export const PROVIDER_IDS = [...CHAT_PROVIDER_ORDER];
export const DEFAULT_CHAT_PROVIDER = "whatsapp";
const WEBSITE_URL = "https://clawd.bot";
const CHAT_PROVIDER_META = {
    telegram: {
        id: "telegram",
        label: "Telegram",
        selectionLabel: "Telegram (Bot API)",
        docsPath: "/telegram",
        docsLabel: "telegram",
        blurb: "simplest way to get started — register a bot with @BotFather and get going.",
        selectionDocsPrefix: "",
        selectionDocsOmitLabel: true,
        selectionExtras: [WEBSITE_URL],
    },
    whatsapp: {
        id: "whatsapp",
        label: "WhatsApp",
        selectionLabel: "WhatsApp (QR link)",
        docsPath: "/whatsapp",
        docsLabel: "whatsapp",
        blurb: "works with your own number; recommend a separate phone + eSIM.",
    },
    discord: {
        id: "discord",
        label: "Discord",
        selectionLabel: "Discord (Bot API)",
        docsPath: "/discord",
        docsLabel: "discord",
        blurb: "very well supported right now.",
    },
    slack: {
        id: "slack",
        label: "Slack",
        selectionLabel: "Slack (Socket Mode)",
        docsPath: "/slack",
        docsLabel: "slack",
        blurb: "supported (Socket Mode).",
    },
    signal: {
        id: "signal",
        label: "Signal",
        selectionLabel: "Signal (signal-cli)",
        docsPath: "/signal",
        docsLabel: "signal",
        blurb: 'signal-cli linked device; more setup (David Reagans: "Hop on Discord.").',
    },
    imessage: {
        id: "imessage",
        label: "iMessage",
        selectionLabel: "iMessage (imsg)",
        docsPath: "/imessage",
        docsLabel: "imessage",
        blurb: "this is still a work in progress.",
    },
    msteams: {
        id: "msteams",
        label: "MS Teams",
        selectionLabel: "Microsoft Teams (Bot Framework)",
        docsPath: "/msteams",
        docsLabel: "msteams",
        blurb: "supported (Bot Framework).",
    },
};
export const CHAT_PROVIDER_ALIASES = {
    imsg: "imessage",
    teams: "msteams",
};
const normalizeProviderKey = (raw) => {
    const normalized = raw?.trim().toLowerCase();
    return normalized || undefined;
};
export function listChatProviders() {
    return CHAT_PROVIDER_ORDER.map((id) => CHAT_PROVIDER_META[id]);
}
export function listChatProviderAliases() {
    return Object.keys(CHAT_PROVIDER_ALIASES);
}
export function getChatProviderMeta(id) {
    return CHAT_PROVIDER_META[id];
}
export function normalizeChatProviderId(raw) {
    const normalized = normalizeProviderKey(raw);
    if (!normalized)
        return null;
    const resolved = CHAT_PROVIDER_ALIASES[normalized] ?? normalized;
    return CHAT_PROVIDER_ORDER.includes(resolved)
        ? resolved
        : null;
}
// Provider docking: prefer this helper in shared code. Importing from
// `src/providers/plugins/*` can eagerly load provider implementations.
export function normalizeProviderId(raw) {
    return normalizeChatProviderId(raw);
}
export function formatProviderPrimerLine(meta) {
    return `${meta.label}: ${meta.blurb}`;
}
export function formatProviderSelectionLine(meta, docsLink) {
    const docsPrefix = meta.selectionDocsPrefix ?? "Docs:";
    const docsLabel = meta.docsLabel ?? meta.id;
    const docs = meta.selectionDocsOmitLabel
        ? docsLink(meta.docsPath)
        : docsLink(meta.docsPath, docsLabel);
    const extras = (meta.selectionExtras ?? []).filter(Boolean).join(" ");
    return `${meta.label} — ${meta.blurb} ${docsPrefix ? `${docsPrefix} ` : ""}${docs}${extras ? ` ${extras}` : ""}`;
}
