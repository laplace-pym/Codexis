function extractLastUserText(input) {
    for (let i = input.length - 1; i >= 0; i -= 1) {
        const item = input[i];
        if (!item || item.role !== "user")
            continue;
        const content = item.content;
        if (Array.isArray(content)) {
            const text = content
                .filter((c) => !!c &&
                typeof c === "object" &&
                c.type === "input_text" &&
                typeof c.text === "string")
                .map((c) => c.text)
                .join("\n")
                .trim();
            if (text)
                return text;
        }
    }
    return "";
}
function extractToolOutput(input) {
    for (const itemRaw of input) {
        const item = itemRaw;
        if (!item || item.type !== "function_call_output")
            continue;
        return typeof item.output === "string" ? item.output : "";
    }
    return "";
}
async function* fakeOpenAIResponsesStream(params) {
    const input = Array.isArray(params.input) ? params.input : [];
    const toolOutput = extractToolOutput(input);
    if (!toolOutput) {
        const prompt = extractLastUserText(input);
        const quoted = /"([^"]+)"/.exec(prompt)?.[1];
        const toolPath = quoted ?? "package.json";
        const argsJson = JSON.stringify({ path: toolPath });
        yield {
            type: "response.output_item.added",
            item: {
                type: "function_call",
                id: "fc_test_1",
                call_id: "call_test_1",
                name: "read",
                arguments: "",
            },
        };
        yield { type: "response.function_call_arguments.delta", delta: argsJson };
        yield {
            type: "response.output_item.done",
            item: {
                type: "function_call",
                id: "fc_test_1",
                call_id: "call_test_1",
                name: "read",
                arguments: argsJson,
            },
        };
        yield {
            type: "response.completed",
            response: {
                status: "completed",
                usage: { input_tokens: 10, output_tokens: 10, total_tokens: 20 },
            },
        };
        return;
    }
    const nonceA = /nonceA=([^\s]+)/.exec(toolOutput)?.[1] ?? "";
    const nonceB = /nonceB=([^\s]+)/.exec(toolOutput)?.[1] ?? "";
    const reply = `${nonceA} ${nonceB}`.trim();
    yield {
        type: "response.output_item.added",
        item: {
            type: "message",
            id: "msg_test_1",
            role: "assistant",
            content: [],
            status: "in_progress",
        },
    };
    yield {
        type: "response.output_item.done",
        item: {
            type: "message",
            id: "msg_test_1",
            role: "assistant",
            status: "completed",
            content: [{ type: "output_text", text: reply, annotations: [] }],
        },
    };
    yield {
        type: "response.completed",
        response: {
            status: "completed",
            usage: { input_tokens: 10, output_tokens: 10, total_tokens: 20 },
        },
    };
}
function decodeBodyText(body) {
    if (!body)
        return "";
    if (typeof body === "string")
        return body;
    if (body instanceof Uint8Array)
        return Buffer.from(body).toString("utf8");
    if (body instanceof ArrayBuffer)
        return Buffer.from(new Uint8Array(body)).toString("utf8");
    return "";
}
async function buildOpenAIResponsesSse(params) {
    const events = [];
    for await (const event of fakeOpenAIResponsesStream(params)) {
        events.push(event);
    }
    const sse = `${events.map((e) => `data: ${JSON.stringify(e)}\n\n`).join("")}data: [DONE]\n\n`;
    const encoder = new TextEncoder();
    const body = new ReadableStream({
        start(controller) {
            controller.enqueue(encoder.encode(sse));
            controller.close();
        },
    });
    return new Response(body, {
        status: 200,
        headers: { "content-type": "text/event-stream" },
    });
}
export function installOpenAiResponsesMock(params) {
    const originalFetch = globalThis.fetch;
    const baseUrl = params?.baseUrl ?? "https://api.openai.com/v1";
    const responsesUrl = `${baseUrl}/responses`;
    const isResponsesRequest = (url) => url === responsesUrl ||
        url.startsWith(`${responsesUrl}/`) ||
        url.startsWith(`${responsesUrl}?`);
    const fetchImpl = async (input, init) => {
        const url = typeof input === "string" ? input : input instanceof URL ? input.toString() : input.url;
        if (isResponsesRequest(url)) {
            const bodyText = typeof init?.body !== "undefined"
                ? decodeBodyText(init.body)
                : input instanceof Request
                    ? await input.clone().text()
                    : "";
            const parsed = bodyText ? JSON.parse(bodyText) : {};
            const inputItems = Array.isArray(parsed.input) ? parsed.input : [];
            return await buildOpenAIResponsesSse({ input: inputItems });
        }
        if (url.startsWith(baseUrl)) {
            throw new Error(`unexpected OpenAI request in mock test: ${url}`);
        }
        if (!originalFetch) {
            throw new Error(`fetch is not available (url=${url})`);
        }
        return await originalFetch(input, init);
    };
    globalThis.fetch = fetchImpl;
    return {
        baseUrl,
        restore: () => {
            globalThis.fetch = originalFetch;
        },
    };
}
