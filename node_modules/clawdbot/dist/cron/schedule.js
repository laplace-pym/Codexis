import { Cron } from "croner";
export function computeNextRunAtMs(schedule, nowMs) {
    if (schedule.kind === "at") {
        return schedule.atMs > nowMs ? schedule.atMs : undefined;
    }
    if (schedule.kind === "every") {
        const everyMs = Math.max(1, Math.floor(schedule.everyMs));
        const anchor = Math.max(0, Math.floor(schedule.anchorMs ?? nowMs));
        if (nowMs < anchor)
            return anchor;
        const elapsed = nowMs - anchor;
        const steps = Math.max(1, Math.floor((elapsed + everyMs - 1) / everyMs));
        return anchor + steps * everyMs;
    }
    const expr = schedule.expr.trim();
    if (!expr)
        return undefined;
    const cron = new Cron(expr, {
        timezone: schedule.tz?.trim() || undefined,
        catch: false,
    });
    const next = cron.nextRun(new Date(nowMs));
    return next ? next.getTime() : undefined;
}
