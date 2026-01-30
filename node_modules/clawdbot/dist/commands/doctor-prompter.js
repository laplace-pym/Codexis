import { confirm, select } from "@clack/prompts";
import { stylePromptHint, stylePromptMessage } from "../terminal/prompt-style.js";
import { guardCancel } from "./onboard-helpers.js";
export function createDoctorPrompter(params) {
    const yes = params.options.yes === true;
    const requestedNonInteractive = params.options.nonInteractive === true;
    const shouldRepair = params.options.repair === true || yes;
    const shouldForce = params.options.force === true;
    const isTty = Boolean(process.stdin.isTTY);
    const nonInteractive = requestedNonInteractive || (!isTty && !yes);
    const canPrompt = isTty && !yes && !nonInteractive;
    const confirmDefault = async (p) => {
        if (nonInteractive)
            return false;
        if (shouldRepair)
            return true;
        if (!canPrompt)
            return Boolean(p.initialValue ?? false);
        return (guardCancel(await confirm({
            ...p,
            message: stylePromptMessage(p.message),
        }), params.runtime) === true);
    };
    return {
        confirm: confirmDefault,
        confirmRepair: async (p) => {
            if (nonInteractive)
                return false;
            return confirmDefault(p);
        },
        confirmAggressive: async (p) => {
            if (nonInteractive)
                return false;
            if (shouldRepair && shouldForce)
                return true;
            if (shouldRepair && !shouldForce)
                return false;
            if (!canPrompt)
                return Boolean(p.initialValue ?? false);
            return (guardCancel(await confirm({
                ...p,
                message: stylePromptMessage(p.message),
            }), params.runtime) === true);
        },
        confirmSkipInNonInteractive: async (p) => {
            if (nonInteractive)
                return false;
            if (shouldRepair)
                return true;
            return confirmDefault(p);
        },
        select: async (p, fallback) => {
            if (!canPrompt || shouldRepair)
                return fallback;
            return guardCancel(await select({
                ...p,
                message: stylePromptMessage(p.message),
                options: p.options.map((opt) => opt.hint === undefined ? opt : { ...opt, hint: stylePromptHint(opt.hint) }),
            }), params.runtime);
        },
        shouldRepair,
        shouldForce,
    };
}
