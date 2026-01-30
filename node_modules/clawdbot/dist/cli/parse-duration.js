export function parseDurationMs(raw, opts) {
    const trimmed = String(raw ?? "")
        .trim()
        .toLowerCase();
    if (!trimmed)
        throw new Error("invalid duration (empty)");
    const m = /^(\d+(?:\.\d+)?)(ms|s|m|h|d)?$/.exec(trimmed);
    if (!m)
        throw new Error(`invalid duration: ${raw}`);
    const value = Number(m[1]);
    if (!Number.isFinite(value) || value < 0) {
        throw new Error(`invalid duration: ${raw}`);
    }
    const unit = (m[2] ?? opts?.defaultUnit ?? "ms");
    const multiplier = unit === "ms"
        ? 1
        : unit === "s"
            ? 1000
            : unit === "m"
                ? 60_000
                : unit === "h"
                    ? 3_600_000
                    : 86_400_000;
    const ms = Math.round(value * multiplier);
    if (!Number.isFinite(ms))
        throw new Error(`invalid duration: ${raw}`);
    return ms;
}
