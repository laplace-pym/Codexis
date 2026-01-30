import crypto from "node:crypto";
import fs from "node:fs/promises";
import fsSync from "node:fs";
import os from "node:os";
import path from "node:path";
import { vi } from "vitest";
import { applyPluginAutoEnable } from "../config/plugin-auto-enable.js";
import { setActivePluginRegistry } from "../plugins/runtime.js";
import { DEFAULT_ACCOUNT_ID } from "../routing/session-key.js";
const createStubOutboundAdapter = (channelId) => ({
    deliveryMode: "direct",
    sendText: async () => ({
        channel: channelId,
        messageId: `${channelId}-msg`,
    }),
    sendMedia: async () => ({
        channel: channelId,
        messageId: `${channelId}-msg`,
    }),
});
const createStubChannelPlugin = (params) => ({
    id: params.id,
    meta: {
        id: params.id,
        label: params.label,
        selectionLabel: params.label,
        docsPath: `/channels/${params.id}`,
        blurb: "test stub.",
    },
    capabilities: { chatTypes: ["direct"] },
    config: {
        listAccountIds: () => [DEFAULT_ACCOUNT_ID],
        resolveAccount: () => ({}),
        isConfigured: async () => false,
    },
    status: {
        buildChannelSummary: async () => ({
            configured: false,
            ...(params.summary ? params.summary : {}),
        }),
    },
    outbound: createStubOutboundAdapter(params.id),
    messaging: {
        normalizeTarget: (raw) => raw,
    },
    gateway: {
        logoutAccount: async () => ({
            cleared: false,
            envToken: false,
            loggedOut: false,
        }),
    },
});
const createStubPluginRegistry = () => ({
    plugins: [],
    tools: [],
    hooks: [],
    typedHooks: [],
    channels: [
        {
            pluginId: "whatsapp",
            source: "test",
            plugin: createStubChannelPlugin({ id: "whatsapp", label: "WhatsApp" }),
        },
        {
            pluginId: "telegram",
            source: "test",
            plugin: createStubChannelPlugin({
                id: "telegram",
                label: "Telegram",
                summary: { tokenSource: "none", lastProbeAt: null },
            }),
        },
        {
            pluginId: "discord",
            source: "test",
            plugin: createStubChannelPlugin({ id: "discord", label: "Discord" }),
        },
        {
            pluginId: "slack",
            source: "test",
            plugin: createStubChannelPlugin({ id: "slack", label: "Slack" }),
        },
        {
            pluginId: "signal",
            source: "test",
            plugin: createStubChannelPlugin({
                id: "signal",
                label: "Signal",
                summary: { lastProbeAt: null },
            }),
        },
        {
            pluginId: "imessage",
            source: "test",
            plugin: createStubChannelPlugin({ id: "imessage", label: "iMessage" }),
        },
        {
            pluginId: "msteams",
            source: "test",
            plugin: createStubChannelPlugin({ id: "msteams", label: "Microsoft Teams" }),
        },
        {
            pluginId: "matrix",
            source: "test",
            plugin: createStubChannelPlugin({ id: "matrix", label: "Matrix" }),
        },
        {
            pluginId: "zalo",
            source: "test",
            plugin: createStubChannelPlugin({ id: "zalo", label: "Zalo" }),
        },
        {
            pluginId: "zalouser",
            source: "test",
            plugin: createStubChannelPlugin({ id: "zalouser", label: "Zalo Personal" }),
        },
        {
            pluginId: "bluebubbles",
            source: "test",
            plugin: createStubChannelPlugin({ id: "bluebubbles", label: "BlueBubbles" }),
        },
    ],
    providers: [],
    gatewayHandlers: {},
    httpHandlers: [],
    httpRoutes: [],
    cliRegistrars: [],
    services: [],
    commands: [],
    diagnostics: [],
});
const hoisted = vi.hoisted(() => ({
    testTailnetIPv4: { value: undefined },
    piSdkMock: {
        enabled: false,
        discoverCalls: 0,
        models: [],
    },
    cronIsolatedRun: vi.fn(async () => ({ status: "ok", summary: "ok" })),
    agentCommand: vi.fn().mockResolvedValue(undefined),
    testIsNixMode: { value: false },
    sessionStoreSaveDelayMs: { value: 0 },
    embeddedRunMock: {
        activeIds: new Set(),
        abortCalls: [],
        waitCalls: [],
        waitResults: new Map(),
    },
    getReplyFromConfig: vi.fn().mockResolvedValue(undefined),
    sendWhatsAppMock: vi.fn().mockResolvedValue({ messageId: "msg-1", toJid: "jid-1" }),
}));
const pluginRegistryState = {
    registry: createStubPluginRegistry(),
};
setActivePluginRegistry(pluginRegistryState.registry);
export const setTestPluginRegistry = (registry) => {
    pluginRegistryState.registry = registry;
    setActivePluginRegistry(registry);
};
export const resetTestPluginRegistry = () => {
    pluginRegistryState.registry = createStubPluginRegistry();
    setActivePluginRegistry(pluginRegistryState.registry);
};
const testConfigRoot = {
    value: path.join(os.tmpdir(), `clawdbot-gateway-test-${process.pid}-${crypto.randomUUID()}`),
};
export const setTestConfigRoot = (root) => {
    testConfigRoot.value = root;
    process.env.CLAWDBOT_CONFIG_PATH = path.join(root, "clawdbot.json");
};
export const testTailnetIPv4 = hoisted.testTailnetIPv4;
export const piSdkMock = hoisted.piSdkMock;
export const cronIsolatedRun = hoisted.cronIsolatedRun;
export const agentCommand = hoisted.agentCommand;
export const getReplyFromConfig = hoisted.getReplyFromConfig;
export const testState = {
    agentConfig: undefined,
    agentsConfig: undefined,
    bindingsConfig: undefined,
    channelsConfig: undefined,
    sessionStorePath: undefined,
    sessionConfig: undefined,
    allowFrom: undefined,
    cronStorePath: undefined,
    cronEnabled: false,
    gatewayBind: undefined,
    gatewayAuth: undefined,
    gatewayControlUi: undefined,
    hooksConfig: undefined,
    canvasHostPort: undefined,
    legacyIssues: [],
    legacyParsed: {},
    migrationConfig: null,
    migrationChanges: [],
};
export const testIsNixMode = hoisted.testIsNixMode;
export const sessionStoreSaveDelayMs = hoisted.sessionStoreSaveDelayMs;
export const embeddedRunMock = hoisted.embeddedRunMock;
vi.mock("@mariozechner/pi-coding-agent", async () => {
    const actual = await vi.importActual("@mariozechner/pi-coding-agent");
    return {
        ...actual,
        discoverModels: (...args) => {
            if (!piSdkMock.enabled) {
                return actual.discoverModels(...args);
            }
            piSdkMock.discoverCalls += 1;
            return piSdkMock.models;
        },
    };
});
vi.mock("../cron/isolated-agent.js", () => ({
    runCronIsolatedAgentTurn: (...args) => cronIsolatedRun(...args),
}));
vi.mock("../infra/tailnet.js", () => ({
    pickPrimaryTailnetIPv4: () => testTailnetIPv4.value,
    pickPrimaryTailnetIPv6: () => undefined,
}));
vi.mock("../config/sessions.js", async () => {
    const actual = await vi.importActual("../config/sessions.js");
    return {
        ...actual,
        saveSessionStore: vi.fn(async (storePath, store) => {
            const delay = sessionStoreSaveDelayMs.value;
            if (delay > 0) {
                await new Promise((resolve) => setTimeout(resolve, delay));
            }
            return actual.saveSessionStore(storePath, store);
        }),
    };
});
vi.mock("../config/config.js", async () => {
    const actual = await vi.importActual("../config/config.js");
    const resolveConfigPath = () => path.join(testConfigRoot.value, "clawdbot.json");
    const hashConfigRaw = (raw) => crypto
        .createHash("sha256")
        .update(raw ?? "")
        .digest("hex");
    const readConfigFileSnapshot = async () => {
        if (testState.legacyIssues.length > 0) {
            const raw = JSON.stringify(testState.legacyParsed ?? {});
            return {
                path: resolveConfigPath(),
                exists: true,
                raw,
                parsed: testState.legacyParsed ?? {},
                valid: false,
                config: {},
                hash: hashConfigRaw(raw),
                issues: testState.legacyIssues.map((issue) => ({
                    path: issue.path,
                    message: issue.message,
                })),
                legacyIssues: testState.legacyIssues,
            };
        }
        const configPath = resolveConfigPath();
        try {
            await fs.access(configPath);
        }
        catch {
            return {
                path: configPath,
                exists: false,
                raw: null,
                parsed: {},
                valid: true,
                config: {},
                hash: hashConfigRaw(null),
                issues: [],
                legacyIssues: [],
            };
        }
        try {
            const raw = await fs.readFile(configPath, "utf-8");
            const parsed = JSON.parse(raw);
            return {
                path: configPath,
                exists: true,
                raw,
                parsed,
                valid: true,
                config: parsed,
                hash: hashConfigRaw(raw),
                issues: [],
                legacyIssues: [],
            };
        }
        catch (err) {
            return {
                path: configPath,
                exists: true,
                raw: null,
                parsed: {},
                valid: false,
                config: {},
                hash: hashConfigRaw(null),
                issues: [{ path: "", message: `read failed: ${String(err)}` }],
                legacyIssues: [],
            };
        }
    };
    const writeConfigFile = vi.fn(async (cfg) => {
        const configPath = resolveConfigPath();
        await fs.mkdir(path.dirname(configPath), { recursive: true });
        const raw = JSON.stringify(cfg, null, 2).trimEnd().concat("\n");
        await fs.writeFile(configPath, raw, "utf-8");
    });
    return {
        ...actual,
        get CONFIG_PATH_CLAWDBOT() {
            return resolveConfigPath();
        },
        get STATE_DIR_CLAWDBOT() {
            return path.dirname(resolveConfigPath());
        },
        get isNixMode() {
            return testIsNixMode.value;
        },
        migrateLegacyConfig: (raw) => ({
            config: testState.migrationConfig ?? raw,
            changes: testState.migrationChanges,
        }),
        loadConfig: () => {
            const configPath = resolveConfigPath();
            let fileConfig = {};
            try {
                if (fsSync.existsSync(configPath)) {
                    const raw = fsSync.readFileSync(configPath, "utf-8");
                    fileConfig = JSON.parse(raw);
                }
            }
            catch {
                fileConfig = {};
            }
            const fileAgents = fileConfig.agents &&
                typeof fileConfig.agents === "object" &&
                !Array.isArray(fileConfig.agents)
                ? fileConfig.agents
                : {};
            const fileDefaults = fileAgents.defaults &&
                typeof fileAgents.defaults === "object" &&
                !Array.isArray(fileAgents.defaults)
                ? fileAgents.defaults
                : {};
            const defaults = {
                model: { primary: "anthropic/claude-opus-4-5" },
                workspace: path.join(os.tmpdir(), "clawd-gateway-test"),
                ...fileDefaults,
                ...testState.agentConfig,
            };
            const agents = testState.agentsConfig
                ? { ...fileAgents, ...testState.agentsConfig, defaults }
                : { ...fileAgents, defaults };
            const fileBindings = Array.isArray(fileConfig.bindings)
                ? fileConfig.bindings
                : undefined;
            const fileChannels = fileConfig.channels &&
                typeof fileConfig.channels === "object" &&
                !Array.isArray(fileConfig.channels)
                ? { ...fileConfig.channels }
                : {};
            const overrideChannels = testState.channelsConfig && typeof testState.channelsConfig === "object"
                ? { ...testState.channelsConfig }
                : {};
            const mergedChannels = { ...fileChannels, ...overrideChannels };
            if (testState.allowFrom !== undefined) {
                const existing = mergedChannels.whatsapp &&
                    typeof mergedChannels.whatsapp === "object" &&
                    !Array.isArray(mergedChannels.whatsapp)
                    ? mergedChannels.whatsapp
                    : {};
                mergedChannels.whatsapp = {
                    ...existing,
                    allowFrom: testState.allowFrom,
                };
            }
            const channels = Object.keys(mergedChannels).length > 0 ? mergedChannels : undefined;
            const fileSession = fileConfig.session &&
                typeof fileConfig.session === "object" &&
                !Array.isArray(fileConfig.session)
                ? fileConfig.session
                : {};
            const session = {
                ...fileSession,
                mainKey: fileSession.mainKey ?? "main",
            };
            if (typeof testState.sessionStorePath === "string")
                session.store = testState.sessionStorePath;
            if (testState.sessionConfig)
                Object.assign(session, testState.sessionConfig);
            const fileGateway = fileConfig.gateway &&
                typeof fileConfig.gateway === "object" &&
                !Array.isArray(fileConfig.gateway)
                ? { ...fileConfig.gateway }
                : {};
            if (testState.gatewayBind)
                fileGateway.bind = testState.gatewayBind;
            if (testState.gatewayAuth)
                fileGateway.auth = testState.gatewayAuth;
            if (testState.gatewayControlUi)
                fileGateway.controlUi = testState.gatewayControlUi;
            const gateway = Object.keys(fileGateway).length > 0 ? fileGateway : undefined;
            const fileCanvasHost = fileConfig.canvasHost &&
                typeof fileConfig.canvasHost === "object" &&
                !Array.isArray(fileConfig.canvasHost)
                ? { ...fileConfig.canvasHost }
                : {};
            if (typeof testState.canvasHostPort === "number")
                fileCanvasHost.port = testState.canvasHostPort;
            const canvasHost = Object.keys(fileCanvasHost).length > 0 ? fileCanvasHost : undefined;
            const hooks = testState.hooksConfig ?? fileConfig.hooks;
            const fileCron = fileConfig.cron && typeof fileConfig.cron === "object" && !Array.isArray(fileConfig.cron)
                ? { ...fileConfig.cron }
                : {};
            if (typeof testState.cronEnabled === "boolean")
                fileCron.enabled = testState.cronEnabled;
            if (typeof testState.cronStorePath === "string")
                fileCron.store = testState.cronStorePath;
            const cron = Object.keys(fileCron).length > 0 ? fileCron : undefined;
            const config = {
                ...fileConfig,
                agents,
                bindings: testState.bindingsConfig ?? fileBindings,
                channels,
                session,
                gateway,
                canvasHost,
                hooks,
                cron,
            };
            return applyPluginAutoEnable({ config, env: process.env }).config;
        },
        parseConfigJson5: (raw) => {
            try {
                return { ok: true, parsed: JSON.parse(raw) };
            }
            catch (err) {
                return { ok: false, error: String(err) };
            }
        },
        validateConfigObject: (parsed) => ({
            ok: true,
            config: parsed,
            issues: [],
        }),
        readConfigFileSnapshot,
        writeConfigFile,
    };
});
vi.mock("../agents/pi-embedded.js", async () => {
    const actual = await vi.importActual("../agents/pi-embedded.js");
    return {
        ...actual,
        isEmbeddedPiRunActive: (sessionId) => embeddedRunMock.activeIds.has(sessionId),
        abortEmbeddedPiRun: (sessionId) => {
            embeddedRunMock.abortCalls.push(sessionId);
            return embeddedRunMock.activeIds.has(sessionId);
        },
        waitForEmbeddedPiRunEnd: async (sessionId) => {
            embeddedRunMock.waitCalls.push(sessionId);
            return embeddedRunMock.waitResults.get(sessionId) ?? true;
        },
    };
});
vi.mock("../commands/health.js", () => ({
    getHealthSnapshot: vi.fn().mockResolvedValue({ ok: true, stub: true }),
}));
vi.mock("../commands/status.js", () => ({
    getStatusSummary: vi.fn().mockResolvedValue({ ok: true }),
}));
vi.mock("../web/outbound.js", () => ({
    sendMessageWhatsApp: (...args) => hoisted.sendWhatsAppMock(...args),
    sendPollWhatsApp: (...args) => hoisted.sendWhatsAppMock(...args),
}));
vi.mock("../channels/web/index.js", async () => {
    const actual = await vi.importActual("../channels/web/index.js");
    return {
        ...actual,
        sendMessageWhatsApp: (...args) => hoisted.sendWhatsAppMock(...args),
    };
});
vi.mock("../commands/agent.js", () => ({
    agentCommand,
}));
vi.mock("../auto-reply/reply.js", () => ({
    getReplyFromConfig,
}));
vi.mock("../cli/deps.js", async () => {
    const actual = await vi.importActual("../cli/deps.js");
    const base = actual.createDefaultDeps();
    return {
        ...actual,
        createDefaultDeps: () => ({
            ...base,
            sendMessageWhatsApp: (...args) => hoisted.sendWhatsAppMock(...args),
        }),
    };
});
process.env.CLAWDBOT_SKIP_CHANNELS = "1";
process.env.CLAWDBOT_SKIP_CRON = "1";
