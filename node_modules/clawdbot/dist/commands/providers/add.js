import { writeConfigFile } from "../../config/config.js";
import { getProviderPlugin, normalizeProviderId, } from "../../providers/plugins/index.js";
import { DEFAULT_ACCOUNT_ID, normalizeAccountId, } from "../../routing/session-key.js";
import { defaultRuntime } from "../../runtime.js";
import { createClackPrompter } from "../../wizard/clack-prompter.js";
import { setupProviders } from "../onboard-providers.js";
import { applyAccountName, applyProviderAccountConfig, } from "./add-mutators.js";
import { providerLabel, requireValidConfig, shouldUseWizard, } from "./shared.js";
export async function providersAddCommand(opts, runtime = defaultRuntime, params) {
    const cfg = await requireValidConfig(runtime);
    if (!cfg)
        return;
    const useWizard = shouldUseWizard(params);
    if (useWizard) {
        const prompter = createClackPrompter();
        let selection = [];
        const accountIds = {};
        await prompter.intro("Provider setup");
        let nextConfig = await setupProviders(cfg, runtime, prompter, {
            allowDisable: false,
            allowSignalInstall: true,
            promptAccountIds: true,
            onSelection: (value) => {
                selection = value;
            },
            onAccountId: (provider, accountId) => {
                accountIds[provider] = accountId;
            },
        });
        if (selection.length === 0) {
            await prompter.outro("No providers selected.");
            return;
        }
        const wantsNames = await prompter.confirm({
            message: "Add display names for these accounts? (optional)",
            initialValue: false,
        });
        if (wantsNames) {
            for (const provider of selection) {
                const accountId = accountIds[provider] ?? DEFAULT_ACCOUNT_ID;
                const plugin = getProviderPlugin(provider);
                const account = plugin?.config.resolveAccount(nextConfig, accountId);
                const snapshot = plugin?.config.describeAccount?.(account, nextConfig);
                const existingName = snapshot?.name ?? account?.name;
                const name = await prompter.text({
                    message: `${provider} account name (${accountId})`,
                    initialValue: existingName,
                });
                if (name?.trim()) {
                    nextConfig = applyAccountName({
                        cfg: nextConfig,
                        provider,
                        accountId,
                        name,
                    });
                }
            }
        }
        await writeConfigFile(nextConfig);
        await prompter.outro("Providers updated.");
        return;
    }
    const provider = normalizeProviderId(opts.provider);
    if (!provider) {
        runtime.error(`Unknown provider: ${String(opts.provider ?? "")}`);
        runtime.exit(1);
        return;
    }
    const plugin = getProviderPlugin(provider);
    if (!plugin?.setup?.applyAccountConfig) {
        runtime.error(`Provider ${provider} does not support add.`);
        runtime.exit(1);
        return;
    }
    const accountId = plugin.setup.resolveAccountId?.({ cfg, accountId: opts.account }) ??
        normalizeAccountId(opts.account);
    const useEnv = opts.useEnv === true;
    const validationError = plugin.setup.validateInput?.({
        cfg,
        accountId,
        input: {
            name: opts.name,
            token: opts.token,
            tokenFile: opts.tokenFile,
            botToken: opts.botToken,
            appToken: opts.appToken,
            signalNumber: opts.signalNumber,
            cliPath: opts.cliPath,
            dbPath: opts.dbPath,
            service: opts.service,
            region: opts.region,
            authDir: opts.authDir,
            httpUrl: opts.httpUrl,
            httpHost: opts.httpHost,
            httpPort: opts.httpPort,
            useEnv,
        },
    });
    if (validationError) {
        runtime.error(validationError);
        runtime.exit(1);
        return;
    }
    const nextConfig = applyProviderAccountConfig({
        cfg,
        provider,
        accountId,
        name: opts.name,
        token: opts.token,
        tokenFile: opts.tokenFile,
        botToken: opts.botToken,
        appToken: opts.appToken,
        signalNumber: opts.signalNumber,
        cliPath: opts.cliPath,
        dbPath: opts.dbPath,
        service: opts.service,
        region: opts.region,
        authDir: opts.authDir,
        httpUrl: opts.httpUrl,
        httpHost: opts.httpHost,
        httpPort: opts.httpPort,
        useEnv,
    });
    await writeConfigFile(nextConfig);
    runtime.log(`Added ${providerLabel(provider)} account "${accountId}".`);
}
