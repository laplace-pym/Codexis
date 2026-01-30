export function createNodeSubscriptionManager() {
    const nodeSubscriptions = new Map();
    const sessionSubscribers = new Map();
    const toPayloadJSON = (payload) => (payload ? JSON.stringify(payload) : null);
    const subscribe = (nodeId, sessionKey) => {
        const normalizedNodeId = nodeId.trim();
        const normalizedSessionKey = sessionKey.trim();
        if (!normalizedNodeId || !normalizedSessionKey)
            return;
        let nodeSet = nodeSubscriptions.get(normalizedNodeId);
        if (!nodeSet) {
            nodeSet = new Set();
            nodeSubscriptions.set(normalizedNodeId, nodeSet);
        }
        if (nodeSet.has(normalizedSessionKey))
            return;
        nodeSet.add(normalizedSessionKey);
        let sessionSet = sessionSubscribers.get(normalizedSessionKey);
        if (!sessionSet) {
            sessionSet = new Set();
            sessionSubscribers.set(normalizedSessionKey, sessionSet);
        }
        sessionSet.add(normalizedNodeId);
    };
    const unsubscribe = (nodeId, sessionKey) => {
        const normalizedNodeId = nodeId.trim();
        const normalizedSessionKey = sessionKey.trim();
        if (!normalizedNodeId || !normalizedSessionKey)
            return;
        const nodeSet = nodeSubscriptions.get(normalizedNodeId);
        nodeSet?.delete(normalizedSessionKey);
        if (nodeSet?.size === 0)
            nodeSubscriptions.delete(normalizedNodeId);
        const sessionSet = sessionSubscribers.get(normalizedSessionKey);
        sessionSet?.delete(normalizedNodeId);
        if (sessionSet?.size === 0)
            sessionSubscribers.delete(normalizedSessionKey);
    };
    const unsubscribeAll = (nodeId) => {
        const normalizedNodeId = nodeId.trim();
        const nodeSet = nodeSubscriptions.get(normalizedNodeId);
        if (!nodeSet)
            return;
        for (const sessionKey of nodeSet) {
            const sessionSet = sessionSubscribers.get(sessionKey);
            sessionSet?.delete(normalizedNodeId);
            if (sessionSet?.size === 0)
                sessionSubscribers.delete(sessionKey);
        }
        nodeSubscriptions.delete(normalizedNodeId);
    };
    const sendToSession = (sessionKey, event, payload, sendEvent) => {
        const normalizedSessionKey = sessionKey.trim();
        if (!normalizedSessionKey || !sendEvent)
            return;
        const subs = sessionSubscribers.get(normalizedSessionKey);
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
        for (const nodeId of nodeSubscriptions.keys()) {
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
        nodeSubscriptions.clear();
        sessionSubscribers.clear();
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
