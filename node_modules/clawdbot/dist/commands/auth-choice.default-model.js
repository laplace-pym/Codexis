export async function applyDefaultModelChoice(params) {
    if (params.setDefaultModel) {
        const next = params.applyDefaultConfig(params.config);
        if (params.noteDefault) {
            await params.prompter.note(`Default model set to ${params.noteDefault}`, "Model configured");
        }
        return { config: next };
    }
    const next = params.applyProviderConfig(params.config);
    await params.noteAgentModel(params.defaultModel);
    return { config: next, agentModelOverride: params.defaultModel };
}
