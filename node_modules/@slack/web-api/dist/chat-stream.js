"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.ChatStreamer = void 0;
class ChatStreamer {
    /**
     * Instantiate a new chat streamer.
     *
     * @description The "constructor" method creates a unique {@link ChatStreamer} instance that keeps track of one chat stream.
     * @example
     * const client = new WebClient(process.env.SLACK_BOT_TOKEN);
     * const logger = new ConsoleLogger();
     * const args = {
     *   channel: "C0123456789",
     *   thread_ts: "1700000001.123456",
     *   recipient_team_id: "T0123456789",
     *   recipient_user_id: "U0123456789",
     * };
     * const streamer = new ChatStreamer(client, logger, args, { buffer_size: 512 });
     * await streamer.append({
     *   markdown_text: "**hello world!**",
     * });
     * await streamer.stop();
     * @see {@link https://docs.slack.dev/reference/methods/chat.startStream}
     * @see {@link https://docs.slack.dev/reference/methods/chat.appendStream}
     * @see {@link https://docs.slack.dev/reference/methods/chat.stopStream}
     */
    constructor(client, logger, args, options) {
        var _a;
        this.buffer = '';
        this.client = client;
        this.logger = logger;
        this.options = {
            buffer_size: (_a = options.buffer_size) !== null && _a !== void 0 ? _a : 256,
        };
        this.state = 'starting';
        this.streamArgs = args;
    }
    /**
     * Append to the stream.
     *
     * @description The "append" method appends to the chat stream being used. This method can be called multiple times. After the stream is stopped this method cannot be called.
     * @example
     * const streamer = client.chatStream({
     *   channel: "C0123456789",
     *   thread_ts: "1700000001.123456",
     *   recipient_team_id: "T0123456789",
     *   recipient_user_id: "U0123456789",
     * });
     * await streamer.append({
     *   markdown_text: "**hello wo",
     * });
     * await streamer.append({
     *   markdown_text: "rld!**",
     * });
     * await streamer.stop();
     * @see {@link https://docs.slack.dev/reference/methods/chat.appendStream}
     */
    append(args) {
        return __awaiter(this, void 0, void 0, function* () {
            if (this.state === 'completed') {
                throw new Error(`failed to append stream: stream state is ${this.state}`);
            }
            if (args.token) {
                this.token = args.token;
            }
            this.buffer += args.markdown_text;
            if (this.buffer.length >= this.options.buffer_size) {
                return yield this.flushBuffer(args);
            }
            const details = {
                bufferLength: this.buffer.length,
                bufferSize: this.options.buffer_size,
                channel: this.streamArgs.channel,
                recipientTeamId: this.streamArgs.recipient_team_id,
                recipientUserId: this.streamArgs.recipient_user_id,
                threadTs: this.streamArgs.thread_ts,
            };
            this.logger.debug(`ChatStreamer appended to buffer: ${JSON.stringify(details)}`);
            return null;
        });
    }
    /**
     * Stop the stream and finalize the message.
     *
     * @description The "stop" method stops the chat stream being used. This method can be called once to end the stream. Additional "blocks" and "metadata" can be provided.
     *
     * @example
     * const streamer = client.chatStream({
     *   channel: "C0123456789",
     *   thread_ts: "1700000001.123456",
     *   recipient_team_id: "T0123456789",
     *   recipient_user_id: "U0123456789",
     * });
     * await streamer.append({
     *   markdown_text: "**hello world!**",
     * });
     * await streamer.stop();
     * @see {@link https://docs.slack.dev/reference/methods/chat.stopStream}
     */
    stop(args) {
        return __awaiter(this, void 0, void 0, function* () {
            if (this.state === 'completed') {
                throw new Error(`failed to stop stream: stream state is ${this.state}`);
            }
            if (args === null || args === void 0 ? void 0 : args.token) {
                this.token = args.token;
            }
            if (args === null || args === void 0 ? void 0 : args.markdown_text) {
                this.buffer += args.markdown_text;
            }
            if (!this.streamTs) {
                const response = yield this.client.chat.startStream(Object.assign(Object.assign({}, this.streamArgs), { token: this.token }));
                if (!response.ts) {
                    throw new Error('failed to stop stream: stream not started');
                }
                this.streamTs = response.ts;
                this.state = 'in_progress';
            }
            const response = yield this.client.chat.stopStream(Object.assign(Object.assign({ token: this.token, channel: this.streamArgs.channel, ts: this.streamTs }, args), { markdown_text: this.buffer }));
            this.state = 'completed';
            return response;
        });
    }
    flushBuffer(args) {
        return __awaiter(this, void 0, void 0, function* () {
            if (!this.streamTs) {
                const response = yield this.client.chat.startStream(Object.assign(Object.assign(Object.assign(Object.assign({}, this.streamArgs), { token: this.token }), args), { markdown_text: this.buffer }));
                this.buffer = '';
                this.streamTs = response.ts;
                this.state = 'in_progress';
                return response;
            }
            const response = yield this.client.chat.appendStream(Object.assign(Object.assign({ token: this.token, channel: this.streamArgs.channel, ts: this.streamTs }, args), { markdown_text: this.buffer }));
            this.buffer = '';
            return response;
        });
    }
}
exports.ChatStreamer = ChatStreamer;
//# sourceMappingURL=chat-stream.js.map