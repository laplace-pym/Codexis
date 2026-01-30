export function createDisabledNodeBridgeServer() {
    return {
        port: 0,
        close: async () => { },
        invoke: async () => {
            throw new Error("bridge disabled in tests");
        },
        sendEvent: () => { },
        listConnected: () => [],
        listeners: [],
    };
}
