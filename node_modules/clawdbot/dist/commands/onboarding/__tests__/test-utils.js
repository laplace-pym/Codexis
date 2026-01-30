import { vi } from "vitest";
export const makeRuntime = (overrides = {}) => ({
    log: vi.fn(),
    error: vi.fn(),
    exit: vi.fn((code) => {
        throw new Error(`exit:${code}`);
    }),
    ...overrides,
});
export const makePrompter = (overrides = {}) => ({
    intro: vi.fn(async () => { }),
    outro: vi.fn(async () => { }),
    note: vi.fn(async () => { }),
    select: vi.fn(async () => "npm"),
    multiselect: vi.fn(async () => []),
    text: vi.fn(async () => ""),
    confirm: vi.fn(async () => false),
    progress: vi.fn(() => ({ update: vi.fn(), stop: vi.fn() })),
    ...overrides,
});
