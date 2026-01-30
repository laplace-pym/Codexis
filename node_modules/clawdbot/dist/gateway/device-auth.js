export function buildDeviceAuthPayload(params) {
    const version = params.version ?? (params.nonce ? "v2" : "v1");
    const scopes = params.scopes.join(",");
    const token = params.token ?? "";
    const base = [
        version,
        params.deviceId,
        params.clientId,
        params.clientMode,
        params.role,
        scopes,
        String(params.signedAtMs),
        token,
    ];
    if (version === "v2") {
        base.push(params.nonce ?? "");
    }
    return base.join("|");
}
