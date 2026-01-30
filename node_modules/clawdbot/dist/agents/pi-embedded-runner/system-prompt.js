import { buildAgentSystemPrompt } from "../system-prompt.js";
import { buildToolSummaryMap } from "../tool-summaries.js";
export function buildEmbeddedSystemPrompt(params) {
    return buildAgentSystemPrompt({
        workspaceDir: params.workspaceDir,
        defaultThinkLevel: params.defaultThinkLevel,
        reasoningLevel: params.reasoningLevel,
        extraSystemPrompt: params.extraSystemPrompt,
        ownerNumbers: params.ownerNumbers,
        reasoningTagHint: params.reasoningTagHint,
        heartbeatPrompt: params.heartbeatPrompt,
        skillsPrompt: params.skillsPrompt,
        docsPath: params.docsPath,
        ttsHint: params.ttsHint,
        workspaceNotes: params.workspaceNotes,
        reactionGuidance: params.reactionGuidance,
        promptMode: params.promptMode,
        runtimeInfo: params.runtimeInfo,
        messageToolHints: params.messageToolHints,
        sandboxInfo: params.sandboxInfo,
        toolNames: params.tools.map((tool) => tool.name),
        toolSummaries: buildToolSummaryMap(params.tools),
        modelAliasLines: params.modelAliasLines,
        userTimezone: params.userTimezone,
        userTime: params.userTime,
        userTimeFormat: params.userTimeFormat,
        contextFiles: params.contextFiles,
    });
}
export function createSystemPromptOverride(systemPrompt) {
    const trimmed = systemPrompt.trim();
    return () => trimmed;
}
