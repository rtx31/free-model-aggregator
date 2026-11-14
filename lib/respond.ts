// Downstream responses: synthesize the client's wire from one normalized provider
// answer {text, inputTokens, outputTokens}. The entry always calls providers
// non-streaming, so these builders own the whole downstream shape, including the
// usage the client checks. Everything is fixed values: no clock, no randomness.
//
// OpenAI JSON:    chat.completion with choices[0].message.content = text,
//                 finish_reason "stop", usage prompt/completion/total.
// OpenAI SSE:     role chunk, one content chunk, finish chunk, usage chunk, [DONE].
//                 (A usage chunk with choices: [] after the finish chunk is accepted.)
// Anthropic JSON: message with content [{type:"text", text}], stop_reason "end_turn",
//                 usage input/output tokens.
// Anthropic SSE:  message_start, content_block_start, content_block_delta(s),
//                 content_block_stop, message_delta (stop_reason + output usage),
//                 message_stop. The provider always answers stop family 0, so the
//                 downstream stop reason is always "stop" / "end_turn".

import type { Json } from './providers.ts';
import type { UpstreamAnswer } from './upstream.ts';

/** The sim's fixed timestamp (2026-01-01T00:00:00Z); the entry's own values match it. */
export const FIXED_CREATED = 1767225600;
const FIXED_MESSAGE_ID = 'msg_sim';
const FIXED_CHAT_ID = 'chatcmpl-sim';

export function openAIJson(model: string, answer: UpstreamAnswer): Json {
  return {
    id: FIXED_CHAT_ID,
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: answer.text },
        finish_reason: 'stop',
        logprobs: null,
      },
    ],
    usage: {
      prompt_tokens: answer.inputTokens,
      completion_tokens: answer.outputTokens,
      total_tokens: answer.inputTokens + answer.outputTokens,
    },
  };
}

export function anthropicJson(model: string, answer: UpstreamAnswer): Json {
  return {
    id: FIXED_MESSAGE_ID,
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: 'end_turn',
    stop_sequence: null,
    usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
  };
}

function sseLine(data: string): string {
  return `data: ${data}\n\n`;
}

function sseEventLine(event: string, data: string): string {
  return `event: ${event}\n${sseLine(data)}`;
}

/** The OpenAI SSE body for one answer: role chunk, content chunk, finish chunk,
 *  usage chunk, [DONE]. Content deltas are concatenated by the client, so one
 *  content chunk carries the whole text. */
export function openAIStreamBody(model: string, answer: UpstreamAnswer): string {
  const head = { id: FIXED_CHAT_ID, object: 'chat.completion.chunk', created: FIXED_CREATED, model };
  const frames: string[] = [];
  frames.push(
    sseLine(
      JSON.stringify({
        ...head,
        choices: [{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null }],
      }),
    ),
  );
  frames.push(
    sseLine(
      JSON.stringify({
        ...head,
        choices: [{ index: 0, delta: { content: answer.text }, finish_reason: null }],
      }),
    ),
  );
  frames.push(
    sseLine(
      JSON.stringify({
        ...head,
        choices: [{ index: 0, delta: {}, finish_reason: 'stop' }],
      }),
    ),
  );
  frames.push(
    sseLine(
      JSON.stringify({
        ...head,
        choices: [],
        usage: {
          prompt_tokens: answer.inputTokens,
          completion_tokens: answer.outputTokens,
          total_tokens: answer.inputTokens + answer.outputTokens,
        },
      }),
    ),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** The Anthropic SSE body for one answer, in the exact event order the client
 *  enforces: message_start, content_block_start, content_block_delta,
 *  content_block_stop, message_delta, message_stop. */
export function anthropicStreamBody(model: string, answer: UpstreamAnswer): string {
  const frames: string[] = [];
  frames.push(
    sseEventLine(
      'message_start',
      JSON.stringify({
        type: 'message_start',
        message: {
          id: FIXED_MESSAGE_ID,
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: answer.inputTokens, output_tokens: 0 },
        },
      }),
    ),
  );
  frames.push(
    sseEventLine(
      'content_block_start',
      JSON.stringify({
        type: 'content_block_start',
        index: 0,
        content_block: { type: 'text', text: '' },
      }),
    ),
  );
  frames.push(
    sseEventLine(
      'content_block_delta',
      JSON.stringify({
        type: 'content_block_delta',
        index: 0,
        delta: { type: 'text_delta', text: answer.text },
      }),
    ),
  );
  frames.push(
    sseEventLine(
      'content_block_stop',
      JSON.stringify({ type: 'content_block_stop', index: 0 }),
    ),
  );
  frames.push(
    sseEventLine(
      'message_delta',
      JSON.stringify({
        type: 'message_delta',
        delta: { stop_reason: 'end_turn', stop_sequence: null },
        usage: { output_tokens: answer.outputTokens },
      }),
    ),
  );
  frames.push(sseEventLine('message_stop', JSON.stringify({ type: 'message_stop' })));
  return frames.join('');
}
