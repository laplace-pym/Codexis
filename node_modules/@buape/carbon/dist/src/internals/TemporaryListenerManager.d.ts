import type { BaseListener } from "../abstracts/BaseListener.js";
import type { Client } from "../classes/Client.js";
export declare class TemporaryListenerManager {
    private client;
    private listeners;
    private defaultTimeout;
    constructor(client: Client, defaultTimeout?: number);
    register(listener: BaseListener, timeoutMs?: number): () => void;
    unregister(id: string | BaseListener): boolean;
    private generateId;
    getCount(): number;
    cleanup(): void;
    getMetrics(): {
        count: number;
        oldestAge: number;
        newestAge: number;
        averageAge: number;
    };
}
//# sourceMappingURL=TemporaryListenerManager.d.ts.map