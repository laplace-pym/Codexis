import { DEFAULT_ACCOUNT_ID, normalizeAccountId } from "../../../routing/session-key.js";
export const promptAccountId = async (params) => {
    const existingIds = params.listAccountIds(params.cfg);
    const initial = params.currentId?.trim() || params.defaultAccountId || DEFAULT_ACCOUNT_ID;
    const choice = (await params.prompter.select({
        message: `${params.label} account`,
        options: [
            ...existingIds.map((id) => ({
                value: id,
                label: id === DEFAULT_ACCOUNT_ID ? "default (primary)" : id,
            })),
            { value: "__new__", label: "Add a new account" },
        ],
        initialValue: initial,
    }));
    if (choice !== "__new__")
        return normalizeAccountId(choice);
    const entered = await params.prompter.text({
        message: `New ${params.label} account id`,
        validate: (value) => (value?.trim() ? undefined : "Required"),
    });
    const normalized = normalizeAccountId(String(entered));
    if (String(entered).trim() !== normalized) {
        await params.prompter.note(`Normalized account id to "${normalized}".`, `${params.label} account`);
    }
    return normalized;
};
export function addWildcardAllowFrom(allowFrom) {
    const next = (allowFrom ?? []).map((v) => String(v).trim()).filter(Boolean);
    if (!next.includes("*"))
        next.push("*");
    return next;
}
