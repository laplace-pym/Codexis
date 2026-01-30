import { Button } from "@buape/carbon";
import { ButtonStyle, Routes } from "discord-api-types/v10";
import { GatewayClient } from "../../gateway/client.js";
import { GATEWAY_CLIENT_MODES, GATEWAY_CLIENT_NAMES } from "../../utils/message-channel.js";
import { createDiscordClient } from "../send.shared.js";
import { logDebug, logError } from "../../logger.js";
const EXEC_APPROVAL_KEY = "execapproval";
function encodeCustomIdValue(value) {
    return encodeURIComponent(value);
}
function decodeCustomIdValue(value) {
    try {
        return decodeURIComponent(value);
    }
    catch {
        return value;
    }
}
export function buildExecApprovalCustomId(approvalId, action) {
    return [`${EXEC_APPROVAL_KEY}:id=${encodeCustomIdValue(approvalId)}`, `action=${action}`].join(";");
}
export function parseExecApprovalData(data) {
    if (!data || typeof data !== "object")
        return null;
    const coerce = (value) => typeof value === "string" || typeof value === "number" ? String(value) : "";
    const rawId = coerce(data.id);
    const rawAction = coerce(data.action);
    if (!rawId || !rawAction)
        return null;
    const action = rawAction;
    if (action !== "allow-once" && action !== "allow-always" && action !== "deny") {
        return null;
    }
    return {
        approvalId: decodeCustomIdValue(rawId),
        action,
    };
}
function formatExecApprovalEmbed(request) {
    const commandText = request.request.command;
    const commandPreview = commandText.length > 1000 ? `${commandText.slice(0, 1000)}...` : commandText;
    const expiresIn = Math.max(0, Math.round((request.expiresAtMs - Date.now()) / 1000));
    const fields = [
        {
            name: "Command",
            value: `\`\`\`\n${commandPreview}\n\`\`\``,
            inline: false,
        },
    ];
    if (request.request.cwd) {
        fields.push({
            name: "Working Directory",
            value: request.request.cwd,
            inline: true,
        });
    }
    if (request.request.host) {
        fields.push({
            name: "Host",
            value: request.request.host,
            inline: true,
        });
    }
    if (request.request.agentId) {
        fields.push({
            name: "Agent",
            value: request.request.agentId,
            inline: true,
        });
    }
    return {
        title: "Exec Approval Required",
        description: "A command needs your approval.",
        color: 0xffa500, // Orange
        fields,
        footer: { text: `Expires in ${expiresIn}s | ID: ${request.id}` },
        timestamp: new Date().toISOString(),
    };
}
function formatResolvedEmbed(request, decision, resolvedBy) {
    const commandText = request.request.command;
    const commandPreview = commandText.length > 500 ? `${commandText.slice(0, 500)}...` : commandText;
    const decisionLabel = decision === "allow-once"
        ? "Allowed (once)"
        : decision === "allow-always"
            ? "Allowed (always)"
            : "Denied";
    const color = decision === "deny" ? 0xed4245 : decision === "allow-always" ? 0x5865f2 : 0x57f287;
    return {
        title: `Exec Approval: ${decisionLabel}`,
        description: resolvedBy ? `Resolved by ${resolvedBy}` : "Resolved",
        color,
        fields: [
            {
                name: "Command",
                value: `\`\`\`\n${commandPreview}\n\`\`\``,
                inline: false,
            },
        ],
        footer: { text: `ID: ${request.id}` },
        timestamp: new Date().toISOString(),
    };
}
function formatExpiredEmbed(request) {
    const commandText = request.request.command;
    const commandPreview = commandText.length > 500 ? `${commandText.slice(0, 500)}...` : commandText;
    return {
        title: "Exec Approval: Expired",
        description: "This approval request has expired.",
        color: 0x99aab5, // Gray
        fields: [
            {
                name: "Command",
                value: `\`\`\`\n${commandPreview}\n\`\`\``,
                inline: false,
            },
        ],
        footer: { text: `ID: ${request.id}` },
        timestamp: new Date().toISOString(),
    };
}
export class DiscordExecApprovalHandler {
    gatewayClient = null;
    pending = new Map();
    requestCache = new Map();
    opts;
    started = false;
    constructor(opts) {
        this.opts = opts;
    }
    shouldHandle(request) {
        const config = this.opts.config;
        if (!config.enabled)
            return false;
        if (!config.approvers || config.approvers.length === 0)
            return false;
        // Check agent filter
        if (config.agentFilter?.length) {
            if (!request.request.agentId)
                return false;
            if (!config.agentFilter.includes(request.request.agentId))
                return false;
        }
        // Check session filter (substring match)
        if (config.sessionFilter?.length) {
            const session = request.request.sessionKey;
            if (!session)
                return false;
            const matches = config.sessionFilter.some((p) => {
                try {
                    return session.includes(p) || new RegExp(p).test(session);
                }
                catch {
                    return session.includes(p);
                }
            });
            if (!matches)
                return false;
        }
        return true;
    }
    async start() {
        if (this.started)
            return;
        this.started = true;
        const config = this.opts.config;
        if (!config.enabled) {
            logDebug("discord exec approvals: disabled");
            return;
        }
        if (!config.approvers || config.approvers.length === 0) {
            logDebug("discord exec approvals: no approvers configured");
            return;
        }
        logDebug("discord exec approvals: starting handler");
        this.gatewayClient = new GatewayClient({
            url: this.opts.gatewayUrl ?? "ws://127.0.0.1:18789",
            clientName: GATEWAY_CLIENT_NAMES.GATEWAY_CLIENT,
            clientDisplayName: "Discord Exec Approvals",
            mode: GATEWAY_CLIENT_MODES.BACKEND,
            scopes: ["operator.approvals"],
            onEvent: (evt) => this.handleGatewayEvent(evt),
            onHelloOk: () => {
                logDebug("discord exec approvals: connected to gateway");
            },
            onConnectError: (err) => {
                logError(`discord exec approvals: connect error: ${err.message}`);
            },
            onClose: (code, reason) => {
                logDebug(`discord exec approvals: gateway closed: ${code} ${reason}`);
            },
        });
        this.gatewayClient.start();
    }
    async stop() {
        if (!this.started)
            return;
        this.started = false;
        // Clear all pending timeouts
        for (const pending of this.pending.values()) {
            clearTimeout(pending.timeoutId);
        }
        this.pending.clear();
        this.requestCache.clear();
        this.gatewayClient?.stop();
        this.gatewayClient = null;
        logDebug("discord exec approvals: stopped");
    }
    handleGatewayEvent(evt) {
        if (evt.event === "exec.approval.requested") {
            const request = evt.payload;
            void this.handleApprovalRequested(request);
        }
        else if (evt.event === "exec.approval.resolved") {
            const resolved = evt.payload;
            void this.handleApprovalResolved(resolved);
        }
    }
    async handleApprovalRequested(request) {
        if (!this.shouldHandle(request))
            return;
        logDebug(`discord exec approvals: received request ${request.id}`);
        this.requestCache.set(request.id, request);
        const { rest, request: discordRequest } = createDiscordClient({ token: this.opts.token, accountId: this.opts.accountId }, this.opts.cfg);
        const embed = formatExecApprovalEmbed(request);
        // Build action rows with buttons
        const components = [
            {
                type: 1, // ACTION_ROW
                components: [
                    {
                        type: 2, // BUTTON
                        style: ButtonStyle.Success,
                        label: "Allow once",
                        custom_id: buildExecApprovalCustomId(request.id, "allow-once"),
                    },
                    {
                        type: 2, // BUTTON
                        style: ButtonStyle.Primary,
                        label: "Always allow",
                        custom_id: buildExecApprovalCustomId(request.id, "allow-always"),
                    },
                    {
                        type: 2, // BUTTON
                        style: ButtonStyle.Danger,
                        label: "Deny",
                        custom_id: buildExecApprovalCustomId(request.id, "deny"),
                    },
                ],
            },
        ];
        const approvers = this.opts.config.approvers ?? [];
        for (const approver of approvers) {
            const userId = String(approver);
            try {
                // Create DM channel
                const dmChannel = (await discordRequest(() => rest.post(Routes.userChannels(), {
                    body: { recipient_id: userId },
                }), "dm-channel"));
                if (!dmChannel?.id) {
                    logError(`discord exec approvals: failed to create DM for user ${userId}`);
                    continue;
                }
                // Send message with embed and buttons
                const message = (await discordRequest(() => rest.post(Routes.channelMessages(dmChannel.id), {
                    body: {
                        embeds: [embed],
                        components,
                    },
                }), "send-approval"));
                if (!message?.id) {
                    logError(`discord exec approvals: failed to send message to user ${userId}`);
                    continue;
                }
                // Set up timeout
                const timeoutMs = Math.max(0, request.expiresAtMs - Date.now());
                const timeoutId = setTimeout(() => {
                    void this.handleApprovalTimeout(request.id);
                }, timeoutMs);
                this.pending.set(request.id, {
                    discordMessageId: message.id,
                    discordChannelId: dmChannel.id,
                    timeoutId,
                });
                logDebug(`discord exec approvals: sent approval ${request.id} to user ${userId}`);
            }
            catch (err) {
                logError(`discord exec approvals: failed to notify user ${userId}: ${String(err)}`);
            }
        }
    }
    async handleApprovalResolved(resolved) {
        const pending = this.pending.get(resolved.id);
        if (!pending)
            return;
        clearTimeout(pending.timeoutId);
        this.pending.delete(resolved.id);
        const request = this.requestCache.get(resolved.id);
        this.requestCache.delete(resolved.id);
        if (!request)
            return;
        logDebug(`discord exec approvals: resolved ${resolved.id} with ${resolved.decision}`);
        await this.updateMessage(pending.discordChannelId, pending.discordMessageId, formatResolvedEmbed(request, resolved.decision, resolved.resolvedBy));
    }
    async handleApprovalTimeout(approvalId) {
        const pending = this.pending.get(approvalId);
        if (!pending)
            return;
        this.pending.delete(approvalId);
        const request = this.requestCache.get(approvalId);
        this.requestCache.delete(approvalId);
        if (!request)
            return;
        logDebug(`discord exec approvals: timeout for ${approvalId}`);
        await this.updateMessage(pending.discordChannelId, pending.discordMessageId, formatExpiredEmbed(request));
    }
    async updateMessage(channelId, messageId, embed) {
        try {
            const { rest, request: discordRequest } = createDiscordClient({ token: this.opts.token, accountId: this.opts.accountId }, this.opts.cfg);
            await discordRequest(() => rest.patch(Routes.channelMessage(channelId, messageId), {
                body: {
                    embeds: [embed],
                    components: [], // Remove buttons
                },
            }), "update-approval");
        }
        catch (err) {
            logError(`discord exec approvals: failed to update message: ${String(err)}`);
        }
    }
    async resolveApproval(approvalId, decision) {
        if (!this.gatewayClient) {
            logError("discord exec approvals: gateway client not connected");
            return false;
        }
        logDebug(`discord exec approvals: resolving ${approvalId} with ${decision}`);
        try {
            await this.gatewayClient.request("exec.approval.resolve", {
                id: approvalId,
                decision,
            });
            logDebug(`discord exec approvals: resolved ${approvalId} successfully`);
            return true;
        }
        catch (err) {
            logError(`discord exec approvals: resolve failed: ${String(err)}`);
            return false;
        }
    }
}
export class ExecApprovalButton extends Button {
    label = "execapproval";
    customId = `${EXEC_APPROVAL_KEY}:seed=1`;
    style = ButtonStyle.Primary;
    ctx;
    constructor(ctx) {
        super();
        this.ctx = ctx;
    }
    async run(interaction, data) {
        const parsed = parseExecApprovalData(data);
        if (!parsed) {
            try {
                await interaction.update({
                    content: "This approval is no longer valid.",
                    components: [],
                });
            }
            catch {
                // Interaction may have expired
            }
            return;
        }
        const decisionLabel = parsed.action === "allow-once"
            ? "Allowed (once)"
            : parsed.action === "allow-always"
                ? "Allowed (always)"
                : "Denied";
        // Update the message immediately to show the decision
        try {
            await interaction.update({
                content: `Submitting decision: **${decisionLabel}**...`,
                components: [], // Remove buttons
            });
        }
        catch {
            // Interaction may have expired, try to continue anyway
        }
        const ok = await this.ctx.handler.resolveApproval(parsed.approvalId, parsed.action);
        if (!ok) {
            try {
                await interaction.followUp({
                    content: "Failed to submit approval decision. The request may have expired or already been resolved.",
                    ephemeral: true,
                });
            }
            catch {
                // Interaction may have expired
            }
        }
        // On success, the handleApprovalResolved event will update the message with the final result
    }
}
export function createExecApprovalButton(ctx) {
    return new ExecApprovalButton(ctx);
}
