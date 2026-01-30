import { formatProviderPrimerLine, formatProviderSelectionLine, getChatProviderMeta, listChatProviders, } from "../providers/registry.js";
import { formatDocsLink } from "../terminal/links.js";
import { getProviderOnboardingAdapter, listProviderOnboardingAdapters, } from "./onboarding/registry.js";
async function noteProviderPrimer(prompter) {
    const providerLines = listChatProviders().map((meta) => formatProviderPrimerLine(meta));
    await prompter.note([
        "DM security: default is pairing; unknown DMs get a pairing code.",
        "Approve with: clawdbot pairing approve <provider> <code>",
        'Public DMs require dmPolicy="open" + allowFrom=["*"].',
        `Docs: ${formatDocsLink("/start/pairing", "start/pairing")}`,
        "",
        ...providerLines,
    ].join("\n"), "How providers work");
}
function resolveQuickstartDefault(statusByProvider) {
    let best = null;
    for (const [provider, status] of statusByProvider) {
        if (status.quickstartScore == null)
            continue;
        if (!best || status.quickstartScore > best.score) {
            best = { provider, score: status.quickstartScore };
        }
    }
    return best?.provider;
}
async function maybeConfigureDmPolicies(params) {
    const { selection, prompter } = params;
    const dmPolicies = selection
        .map((provider) => getProviderOnboardingAdapter(provider)?.dmPolicy)
        .filter(Boolean);
    if (dmPolicies.length === 0)
        return params.cfg;
    const wants = await prompter.confirm({
        message: "Configure DM access policies now? (default: pairing)",
        initialValue: false,
    });
    if (!wants)
        return params.cfg;
    let cfg = params.cfg;
    const selectPolicy = async (policy) => {
        await prompter.note([
            "Default: pairing (unknown DMs get a pairing code).",
            `Approve: clawdbot pairing approve ${policy.provider} <code>`,
            `Public DMs: ${policy.policyKey}="open" + ${policy.allowFromKey} includes "*".`,
            `Docs: ${formatDocsLink("/start/pairing", "start/pairing")}`,
        ].join("\n"), `${policy.label} DM access`);
        return (await prompter.select({
            message: `${policy.label} DM policy`,
            options: [
                { value: "pairing", label: "Pairing (recommended)" },
                { value: "open", label: "Open (public inbound DMs)" },
                { value: "disabled", label: "Disabled (ignore DMs)" },
            ],
        }));
    };
    for (const policy of dmPolicies) {
        const current = policy.getCurrent(cfg);
        const nextPolicy = await selectPolicy(policy);
        if (nextPolicy !== current) {
            cfg = policy.setPolicy(cfg, nextPolicy);
        }
    }
    return cfg;
}
// Provider-specific prompts moved into onboarding adapters.
export async function setupProviders(cfg, runtime, prompter, options) {
    const forceAllowFromProviders = new Set(options?.forceAllowFromProviders ?? []);
    const accountOverrides = {
        ...options?.accountIds,
    };
    if (options?.whatsappAccountId?.trim()) {
        accountOverrides.whatsapp = options.whatsappAccountId.trim();
    }
    const statusEntries = await Promise.all(listProviderOnboardingAdapters().map((adapter) => adapter.getStatus({ cfg, options, accountOverrides })));
    const statusByProvider = new Map(statusEntries.map((entry) => [entry.provider, entry]));
    const statusLines = statusEntries.flatMap((entry) => entry.statusLines);
    if (statusLines.length > 0) {
        await prompter.note(statusLines.join("\n"), "Provider status");
    }
    const shouldConfigure = options?.skipConfirm
        ? true
        : await prompter.confirm({
            message: "Configure chat providers now?",
            initialValue: true,
        });
    if (!shouldConfigure)
        return cfg;
    await noteProviderPrimer(prompter);
    const selectionOptions = listChatProviders().map((meta) => {
        const status = statusByProvider.get(meta.id);
        return {
            value: meta.id,
            label: meta.selectionLabel,
            ...(status?.selectionHint ? { hint: status.selectionHint } : {}),
        };
    });
    const quickstartDefault = options?.initialSelection?.[0] ??
        resolveQuickstartDefault(statusByProvider);
    let selection;
    if (options?.quickstartDefaults) {
        const choice = (await prompter.select({
            message: "Select provider (QuickStart)",
            options: [
                ...selectionOptions,
                {
                    value: "__skip__",
                    label: "Skip for now",
                    hint: "You can add providers later via `clawdbot providers add`",
                },
            ],
            initialValue: quickstartDefault,
        }));
        selection = choice === "__skip__" ? [] : [choice];
    }
    else {
        const initialSelection = options?.initialSelection ?? [];
        selection = (await prompter.multiselect({
            message: "Select providers (Space to toggle, Enter to continue)",
            options: selectionOptions,
            initialValues: initialSelection.length ? initialSelection : undefined,
        }));
    }
    options?.onSelection?.(selection);
    const selectionNotes = new Map(listChatProviders().map((meta) => [
        meta.id,
        formatProviderSelectionLine(meta, formatDocsLink),
    ]));
    const selectedLines = selection
        .map((provider) => selectionNotes.get(provider))
        .filter((line) => Boolean(line));
    if (selectedLines.length > 0) {
        await prompter.note(selectedLines.join("\n"), "Selected providers");
    }
    const shouldPromptAccountIds = options?.promptAccountIds === true;
    const recordAccount = (provider, accountId) => {
        options?.onAccountId?.(provider, accountId);
        const adapter = getProviderOnboardingAdapter(provider);
        adapter?.onAccountRecorded?.(accountId, options);
    };
    let next = cfg;
    for (const provider of selection) {
        const adapter = getProviderOnboardingAdapter(provider);
        if (!adapter)
            continue;
        const result = await adapter.configure({
            cfg: next,
            runtime,
            prompter,
            options,
            accountOverrides,
            shouldPromptAccountIds,
            forceAllowFrom: forceAllowFromProviders.has(provider),
        });
        next = result.cfg;
        if (result.accountId) {
            recordAccount(provider, result.accountId);
        }
    }
    if (!options?.skipDmPolicyPrompt) {
        next = await maybeConfigureDmPolicies({ cfg: next, selection, prompter });
    }
    if (options?.allowDisable) {
        for (const [providerId, status] of statusByProvider) {
            if (selection.includes(providerId))
                continue;
            if (!status.configured)
                continue;
            const adapter = getProviderOnboardingAdapter(providerId);
            if (!adapter?.disable)
                continue;
            const meta = getChatProviderMeta(providerId);
            const disable = await prompter.confirm({
                message: `Disable ${meta.label} provider?`,
                initialValue: false,
            });
            if (disable) {
                next = adapter.disable(next);
            }
        }
    }
    return next;
}
