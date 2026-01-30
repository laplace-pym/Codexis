import { randomUUID } from "node:crypto";
export function createInMemorySessionStore() {
    const sessions = new Map();
    const runIdToSessionId = new Map();
    const createSession = (params) => {
        const sessionId = params.sessionId ?? randomUUID();
        const session = {
            sessionId,
            sessionKey: params.sessionKey,
            cwd: params.cwd,
            createdAt: Date.now(),
            abortController: null,
            activeRunId: null,
        };
        sessions.set(sessionId, session);
        return session;
    };
    const getSession = (sessionId) => sessions.get(sessionId);
    const getSessionByRunId = (runId) => {
        const sessionId = runIdToSessionId.get(runId);
        return sessionId ? sessions.get(sessionId) : undefined;
    };
    const setActiveRun = (sessionId, runId, abortController) => {
        const session = sessions.get(sessionId);
        if (!session)
            return;
        session.activeRunId = runId;
        session.abortController = abortController;
        runIdToSessionId.set(runId, sessionId);
    };
    const clearActiveRun = (sessionId) => {
        const session = sessions.get(sessionId);
        if (!session)
            return;
        if (session.activeRunId)
            runIdToSessionId.delete(session.activeRunId);
        session.activeRunId = null;
        session.abortController = null;
    };
    const cancelActiveRun = (sessionId) => {
        const session = sessions.get(sessionId);
        if (!session?.abortController)
            return false;
        session.abortController.abort();
        if (session.activeRunId)
            runIdToSessionId.delete(session.activeRunId);
        session.abortController = null;
        session.activeRunId = null;
        return true;
    };
    const clearAllSessionsForTest = () => {
        for (const session of sessions.values()) {
            session.abortController?.abort();
        }
        sessions.clear();
        runIdToSessionId.clear();
    };
    return {
        createSession,
        getSession,
        getSessionByRunId,
        setActiveRun,
        clearActiveRun,
        cancelActiveRun,
        clearAllSessionsForTest,
    };
}
export const defaultAcpSessionStore = createInMemorySessionStore();
