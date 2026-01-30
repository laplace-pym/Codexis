function isOpenAiCompletionsModel(model) {
    return model.api === "openai-completions";
}
export function normalizeModelCompat(model) {
    const baseUrl = model.baseUrl ?? "";
    const isZai = model.provider === "zai" || baseUrl.includes("api.z.ai");
    if (!isZai || !isOpenAiCompletionsModel(model))
        return model;
    const openaiModel = model;
    const compat = openaiModel.compat ?? undefined;
    if (compat?.supportsDeveloperRole === false)
        return model;
    openaiModel.compat = compat
        ? { ...compat, supportsDeveloperRole: false }
        : { supportsDeveloperRole: false };
    return openaiModel;
}
