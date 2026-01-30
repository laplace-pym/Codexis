export function createBridgeSubscriptionManager() {
    const bridgeNodeSubscriptions = new Map();
    const bridgeSessionSubscribers = new Map();
    const toPayloadJSON = (payload) => (payload ? JSON.stringify(payload) : null);
    const subscribe = (nodeId, sessionKey) => {
        const normalizedNodeId = nodeId.trim();
        const normalizedSessionKey = sessionKey.trim();
        if (!normalizedNodeId || !normalizedSessionKey)
            return;
        let nodeSet = bridgeNodeSubscriptions.get(normalizedNodeId);
        if (!nodeSet) {
            nodeSet = new Set();
            bridgeNodeSubscriptions.set(normalizedNodeId, nodeSet);
        }
        if (nodeSet.has(normalizedSessionKey))
            return;
        nodeSet.add(normalizedSessionKey);
        let sessionSet = bridgeSessionSubscribers.get(normalizedSessionKey);
        if (!sessionSet) {
            sessionSet = new Set();
            bridgeSessionSubscribers.set(normalizedSessionKey, sessionSet);
        }
        sessionSet.add(normalizedNodeId);
    };
    const unsubscribe = (nodeId, sessionKey) => {
        const normalizedNodeId = nodeId.trim();
        const normalizedSessionKey = sessionKey.trim();
        if (!normalizedNodeId || !normalizedSessionKey)
            return;
        const nodeSet = bridgeNodeSubscriptions.get(normalizedNodeId);
        nodeSet?.delete(normalizedSessionKey);
        if (nodeSet?.size === 0)
            bridgeNodeSubscriptions.delete(normalizedNodeId);
        const sessionSet = bridgeSessionSubscribers.get(normalizedSessionKey);
        sessionSet?.delete(normalizedNodeId);
        if (sessionSet?.size === 0)
            bridgeSessionSubscribers.delete(normalizedSessionKey);
    };
    const unsubscribeAll = (nodeId) => {
        const normalizedNodeId = nodeId.trim();
        const nodeSet = bridgeNodeSubscriptions.get(normalizedNodeId);
        if (!nodeSet)
            return;
        for (const sessionKey of nodeSet) {
            const sessionSet = bridgeSessionSubscribers.get(sessionKey);
            sessionSet?.delete(normalizedNodeId);
            if (sessionSet?.size === 0)
                bridgeSessionSubscribers.delete(sessionKey);
        }
        bridgeNodeSubscriptions.delete(normalizedNodeId);
    };
    const sendToSession = (sessionKey, event, payload, sendEvent) => {
        const normalizedSessionKey = sessionKey.trim();
        if (!normalizedSessionKey || !sendEvent)
            return;
        const subs = bridgeSessionSubscribers.get(normalizedSessionKey);
        if (!subs || subs.size === 0)
            return;
        const payloadJSON = toPayloadJSON(payload);
        for (const nodeId of subs) {
            sendEvent({ nodeId, event, payloadJSON });
        }
    };
    const sendToAllSubscribed = (event, payload, sendEvent) => {
        if (!sendEvent)
            return;
        const payloadJSON = toPayloadJSON(payload);
        for (const nodeId of bridgeNodeSubscriptions.keys()) {
            sendEvent({ nodeId, event, payloadJSON });
        }
    };
    const sendToAllConnected = (event, payload, listConnected, sendEvent) => {
        if (!sendEvent || !listConnected)
            return;
        const payloadJSON = toPayloadJSON(payload);
        for (const node of listConnected()) {
            sendEvent({ nodeId: node.nodeId, event, payloadJSON });
        }
    };
    const clear = () => {
        bridgeNodeSubscriptions.clear();
        bridgeSessionSubscribers.clear();
    };
    return {
        subscribe,
        unsubscribe,
        unsubscribeAll,
        sendToSession,
        sendToAllSubscribed,
        sendToAllConnected,
        clear,
    };
}
