import crypto from "node:crypto";
import net from "node:net";
export async function requestExecHostViaSocket(params) {
    const { socketPath, token, request } = params;
    if (!socketPath || !token)
        return null;
    const timeoutMs = params.timeoutMs ?? 20_000;
    return await new Promise((resolve) => {
        const client = new net.Socket();
        let settled = false;
        let buffer = "";
        const finish = (value) => {
            if (settled)
                return;
            settled = true;
            try {
                client.destroy();
            }
            catch {
                // ignore
            }
            resolve(value);
        };
        const requestJson = JSON.stringify(request);
        const nonce = crypto.randomBytes(16).toString("hex");
        const ts = Date.now();
        const hmac = crypto
            .createHmac("sha256", token)
            .update(`${nonce}:${ts}:${requestJson}`)
            .digest("hex");
        const payload = JSON.stringify({
            type: "exec",
            id: crypto.randomUUID(),
            nonce,
            ts,
            hmac,
            requestJson,
        });
        const timer = setTimeout(() => finish(null), timeoutMs);
        client.on("error", () => finish(null));
        client.connect(socketPath, () => {
            client.write(`${payload}\n`);
        });
        client.on("data", (data) => {
            buffer += data.toString("utf8");
            let idx = buffer.indexOf("\n");
            while (idx !== -1) {
                const line = buffer.slice(0, idx).trim();
                buffer = buffer.slice(idx + 1);
                idx = buffer.indexOf("\n");
                if (!line)
                    continue;
                try {
                    const msg = JSON.parse(line);
                    if (msg?.type === "exec-res") {
                        clearTimeout(timer);
                        if (msg.ok === true && msg.payload) {
                            finish({ ok: true, payload: msg.payload });
                            return;
                        }
                        if (msg.ok === false && msg.error) {
                            finish({ ok: false, error: msg.error });
                            return;
                        }
                        finish(null);
                        return;
                    }
                }
                catch {
                    // ignore
                }
            }
        });
    });
}
