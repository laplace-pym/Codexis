import { resolveBrowserControlUrl } from "../browser/client.js";
import { browserSetDevice, browserSetGeolocation, browserSetHeaders, browserSetHttpCredentials, browserSetLocale, browserSetMedia, browserSetOffline, browserSetTimezone, } from "../browser/client-actions.js";
import { browserAct } from "../browser/client-actions-core.js";
import { danger } from "../globals.js";
import { defaultRuntime } from "../runtime.js";
import { parseBooleanValue } from "../utils/boolean.js";
import { registerBrowserCookiesAndStorageCommands } from "./browser-cli-state.cookies-storage.js";
import { runCommandWithRuntime } from "./cli-utils.js";
function parseOnOff(raw) {
    const parsed = parseBooleanValue(raw);
    return parsed === undefined ? null : parsed;
}
function runBrowserCommand(action) {
    return runCommandWithRuntime(defaultRuntime, action, (err) => {
        defaultRuntime.error(danger(String(err)));
        defaultRuntime.exit(1);
    });
}
export function registerBrowserStateCommands(browser, parentOpts) {
    registerBrowserCookiesAndStorageCommands(browser, parentOpts);
    const set = browser.command("set").description("Browser environment settings");
    set
        .command("viewport")
        .description("Set viewport size (alias for resize)")
        .argument("<width>", "Viewport width", (v) => Number(v))
        .argument("<height>", "Viewport height", (v) => Number(v))
        .option("--target-id <id>", "CDP target id (or unique prefix)")
        .action(async (width, height, opts, cmd) => {
        const parent = parentOpts(cmd);
        const baseUrl = resolveBrowserControlUrl(parent?.url);
        const profile = parent?.browserProfile;
        if (!Number.isFinite(width) || !Number.isFinite(height)) {
            defaultRuntime.error(danger("width and height must be numbers"));
            defaultRuntime.exit(1);
            return;
        }
        await runBrowserCommand(async () => {
            const result = await browserAct(baseUrl, {
                kind: "resize",
                width,
                height,
                targetId: opts.targetId?.trim() || undefined,
            }, { profile });
            if (parent?.json) {
                defaultRuntime.log(JSON.stringify(result, null, 2));
                return;
            }
            defaultRuntime.log(`viewport set: ${width}x${height}`);
        });
    });
    set
        .command("offline")
        .description("Toggle offline mode")
        .argument("<on|off>", "on/off")
        .option("--target-id <id>", "CDP target id (or unique prefix)")
        .action(async (value, opts, cmd) => {
        const parent = parentOpts(cmd);
        const baseUrl = resolveBrowserControlUrl(parent?.url);
        const profile = parent?.browserProfile;
        const offline = parseOnOff(value);
        if (offline === null) {
            defaultRuntime.error(danger("Expected on|off"));
            defaultRuntime.exit(1);
            return;
        }
        await runBrowserCommand(async () => {
            const result = await browserSetOffline(baseUrl, {
                offline,
                targetId: opts.targetId?.trim() || undefined,
                profile,
            });
            if (parent?.json) {
                defaultRuntime.log(JSON.stringify(result, null, 2));
                return;
            }
            defaultRuntime.log(`offline: ${offline}`);
        });
    });
    set
        .command("headers")
        .description("Set extra HTTP headers (JSON object)")
        .requiredOption("--json <json>", "JSON object of headers")
        .option("--target-id <id>", "CDP target id (or unique prefix)")
        .action(async (opts, cmd) => {
        const parent = parentOpts(cmd);
        const baseUrl = resolveBrowserControlUrl(parent?.url);
        const profile = parent?.browserProfile;
        await runBrowserCommand(async () => {
            const parsed = JSON.parse(String(opts.json));
            if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
                throw new Error("headers json must be an object");
            }
            const headers = {};
            for (const [k, v] of Object.entries(parsed)) {
                if (typeof v === "string")
                    headers[k] = v;
            }
            const result = await browserSetHeaders(baseUrl, {
                headers,
                targetId: opts.targetId?.trim() || undefined,
                profile,
            });
            if (parent?.json) {
                defaultRuntime.log(JSON.stringify(result, null, 2));
                return;
            }
            defaultRuntime.log("headers set");
        });
    });
    set
        .command("credentials")
        .description("Set HTTP basic auth credentials")
        .option("--clear", "Clear credentials", false)
        .argument("[username]", "Username")
        .argument("[password]", "Password")
        .option("--target-id <id>", "CDP target id (or unique prefix)")
        .action(async (username, password, opts, cmd) => {
        const parent = parentOpts(cmd);
        const baseUrl = resolveBrowserControlUrl(parent?.url);
        const profile = parent?.browserProfile;
        await runBrowserCommand(async () => {
            const result = await browserSetHttpCredentials(baseUrl, {
                username: username?.trim() || undefined,
                password,
                clear: Boolean(opts.clear),
                targetId: opts.targetId?.trim() || undefined,
                profile,
            });
            if (parent?.json) {
                defaultRuntime.log(JSON.stringify(result, null, 2));
                return;
            }
            defaultRuntime.log(opts.clear ? "credentials cleared" : "credentials set");
        });
    });
    set
        .command("geo")
        .description("Set geolocation (and grant permission)")
        .option("--clear", "Clear geolocation + permissions", false)
        .argument("[latitude]", "Latitude", (v) => Number(v))
        .argument("[longitude]", "Longitude", (v) => Number(v))
        .option("--accuracy <m>", "Accuracy in meters", (v) => Number(v))
        .option("--origin <origin>", "Origin to grant permissions for")
        .option("--target-id <id>", "CDP target id (or unique prefix)")
        .action(async (latitude, longitude, opts, cmd) => {
        const parent = parentOpts(cmd);
        const baseUrl = resolveBrowserControlUrl(parent?.url);
        const profile = parent?.browserProfile;
        await runBrowserCommand(async () => {
            const result = await browserSetGeolocation(baseUrl, {
                latitude: Number.isFinite(latitude) ? latitude : undefined,
                longitude: Number.isFinite(longitude) ? longitude : undefined,
                accuracy: Number.isFinite(opts.accuracy) ? opts.accuracy : undefined,
                origin: opts.origin?.trim() || undefined,
                clear: Boolean(opts.clear),
                targetId: opts.targetId?.trim() || undefined,
                profile,
            });
            if (parent?.json) {
                defaultRuntime.log(JSON.stringify(result, null, 2));
                return;
            }
            defaultRuntime.log(opts.clear ? "geolocation cleared" : "geolocation set");
        });
    });
    set
        .command("media")
        .description("Emulate prefers-color-scheme")
        .argument("<dark|light|none>", "dark/light/none")
        .option("--target-id <id>", "CDP target id (or unique prefix)")
        .action(async (value, opts, cmd) => {
        const parent = parentOpts(cmd);
        const baseUrl = resolveBrowserControlUrl(parent?.url);
        const profile = parent?.browserProfile;
        const v = value.trim().toLowerCase();
        const colorScheme = v === "dark" ? "dark" : v === "light" ? "light" : v === "none" ? "none" : null;
        if (!colorScheme) {
            defaultRuntime.error(danger("Expected dark|light|none"));
            defaultRuntime.exit(1);
            return;
        }
        await runBrowserCommand(async () => {
            const result = await browserSetMedia(baseUrl, {
                colorScheme,
                targetId: opts.targetId?.trim() || undefined,
                profile,
            });
            if (parent?.json) {
                defaultRuntime.log(JSON.stringify(result, null, 2));
                return;
            }
            defaultRuntime.log(`media colorScheme: ${colorScheme}`);
        });
    });
    set
        .command("timezone")
        .description("Override timezone (CDP)")
        .argument("<timezoneId>", "Timezone ID (e.g. America/New_York)")
        .option("--target-id <id>", "CDP target id (or unique prefix)")
        .action(async (timezoneId, opts, cmd) => {
        const parent = parentOpts(cmd);
        const baseUrl = resolveBrowserControlUrl(parent?.url);
        const profile = parent?.browserProfile;
        await runBrowserCommand(async () => {
            const result = await browserSetTimezone(baseUrl, {
                timezoneId,
                targetId: opts.targetId?.trim() || undefined,
                profile,
            });
            if (parent?.json) {
                defaultRuntime.log(JSON.stringify(result, null, 2));
                return;
            }
            defaultRuntime.log(`timezone: ${timezoneId}`);
        });
    });
    set
        .command("locale")
        .description("Override locale (CDP)")
        .argument("<locale>", "Locale (e.g. en-US)")
        .option("--target-id <id>", "CDP target id (or unique prefix)")
        .action(async (locale, opts, cmd) => {
        const parent = parentOpts(cmd);
        const baseUrl = resolveBrowserControlUrl(parent?.url);
        const profile = parent?.browserProfile;
        await runBrowserCommand(async () => {
            const result = await browserSetLocale(baseUrl, {
                locale,
                targetId: opts.targetId?.trim() || undefined,
                profile,
            });
            if (parent?.json) {
                defaultRuntime.log(JSON.stringify(result, null, 2));
                return;
            }
            defaultRuntime.log(`locale: ${locale}`);
        });
    });
    set
        .command("device")
        .description('Apply a Playwright device descriptor (e.g. "iPhone 14")')
        .argument("<name>", "Device name (Playwright devices)")
        .option("--target-id <id>", "CDP target id (or unique prefix)")
        .action(async (name, opts, cmd) => {
        const parent = parentOpts(cmd);
        const baseUrl = resolveBrowserControlUrl(parent?.url);
        const profile = parent?.browserProfile;
        await runBrowserCommand(async () => {
            const result = await browserSetDevice(baseUrl, {
                name,
                targetId: opts.targetId?.trim() || undefined,
                profile,
            });
            if (parent?.json) {
                defaultRuntime.log(JSON.stringify(result, null, 2));
                return;
            }
            defaultRuntime.log(`device: ${name}`);
        });
    });
}
