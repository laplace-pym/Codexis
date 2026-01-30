import { listChannelPlugins } from "../../channels/plugins/index.js";
const CHANNEL_ONBOARDING_ADAPTERS = () => new Map(listChannelPlugins()
    .map((plugin) => plugin.onboarding ? [plugin.id, plugin.onboarding] : null)
    .filter((entry) => Boolean(entry)));
export function getChannelOnboardingAdapter(channel) {
    return CHANNEL_ONBOARDING_ADAPTERS().get(channel);
}
export function listChannelOnboardingAdapters() {
    return Array.from(CHANNEL_ONBOARDING_ADAPTERS().values());
}
// Legacy aliases (pre-rename).
export const getProviderOnboardingAdapter = getChannelOnboardingAdapter;
export const listProviderOnboardingAdapters = listChannelOnboardingAdapters;
