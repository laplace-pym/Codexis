import { ErrorCodes, errorShape, formatValidationErrors, validateExecApprovalRequestParams, validateExecApprovalResolveParams, } from "../protocol/index.js";
export function createExecApprovalHandlers(manager, opts) {
    return {
        "exec.approval.request": async ({ params, respond, context }) => {
            if (!validateExecApprovalRequestParams(params)) {
                respond(false, undefined, errorShape(ErrorCodes.INVALID_REQUEST, `invalid exec.approval.request params: ${formatValidationErrors(validateExecApprovalRequestParams.errors)}`));
                return;
            }
            const p = params;
            const timeoutMs = typeof p.timeoutMs === "number" ? p.timeoutMs : 120_000;
            const explicitId = typeof p.id === "string" && p.id.trim().length > 0 ? p.id.trim() : null;
            if (explicitId && manager.getSnapshot(explicitId)) {
                respond(false, undefined, errorShape(ErrorCodes.INVALID_REQUEST, "approval id already pending"));
                return;
            }
            const request = {
                command: p.command,
                cwd: p.cwd ?? null,
                host: p.host ?? null,
                security: p.security ?? null,
                ask: p.ask ?? null,
                agentId: p.agentId ?? null,
                resolvedPath: p.resolvedPath ?? null,
                sessionKey: p.sessionKey ?? null,
            };
            const record = manager.create(request, timeoutMs, explicitId);
            const decisionPromise = manager.waitForDecision(record, timeoutMs);
            context.broadcast("exec.approval.requested", {
                id: record.id,
                request: record.request,
                createdAtMs: record.createdAtMs,
                expiresAtMs: record.expiresAtMs,
            }, { dropIfSlow: true });
            void opts?.forwarder
                ?.handleRequested({
                id: record.id,
                request: record.request,
                createdAtMs: record.createdAtMs,
                expiresAtMs: record.expiresAtMs,
            })
                .catch((err) => {
                context.logGateway?.error?.(`exec approvals: forward request failed: ${String(err)}`);
            });
            const decision = await decisionPromise;
            respond(true, {
                id: record.id,
                decision,
                createdAtMs: record.createdAtMs,
                expiresAtMs: record.expiresAtMs,
            }, undefined);
        },
        "exec.approval.resolve": async ({ params, respond, client, context }) => {
            if (!validateExecApprovalResolveParams(params)) {
                respond(false, undefined, errorShape(ErrorCodes.INVALID_REQUEST, `invalid exec.approval.resolve params: ${formatValidationErrors(validateExecApprovalResolveParams.errors)}`));
                return;
            }
            const p = params;
            const decision = p.decision;
            if (decision !== "allow-once" && decision !== "allow-always" && decision !== "deny") {
                respond(false, undefined, errorShape(ErrorCodes.INVALID_REQUEST, "invalid decision"));
                return;
            }
            const resolvedBy = client?.connect?.client?.displayName ?? client?.connect?.client?.id;
            const ok = manager.resolve(p.id, decision, resolvedBy ?? null);
            if (!ok) {
                respond(false, undefined, errorShape(ErrorCodes.INVALID_REQUEST, "unknown approval id"));
                return;
            }
            context.broadcast("exec.approval.resolved", { id: p.id, decision, resolvedBy, ts: Date.now() }, { dropIfSlow: true });
            void opts?.forwarder
                ?.handleResolved({ id: p.id, decision, resolvedBy, ts: Date.now() })
                .catch((err) => {
                context.logGateway?.error?.(`exec approvals: forward resolve failed: ${String(err)}`);
            });
            respond(true, { ok: true }, undefined);
        },
    };
}
