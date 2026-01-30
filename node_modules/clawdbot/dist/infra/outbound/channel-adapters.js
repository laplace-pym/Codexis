const DEFAULT_ADAPTER = {
    supportsEmbeds: false,
};
const DISCORD_ADAPTER = {
    supportsEmbeds: true,
    buildCrossContextEmbeds: (originLabel) => [
        {
            description: `From ${originLabel}`,
        },
    ],
};
export function getChannelMessageAdapter(channel) {
    if (channel === "discord")
        return DISCORD_ADAPTER;
    return DEFAULT_ADAPTER;
}
