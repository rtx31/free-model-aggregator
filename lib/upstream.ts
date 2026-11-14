// Upstream providers: extract a wire-neutral request from either client wire,
// build the provider's own request in its wire, and parse its answer back into
// one normalized shape. Everything here is deterministic: no clock, no randomness.
//
// The three provider wires (see fma-spec.md §2):
//   openai    POST <base>/v1/chat/completions
//   anthropic POST <base>/v1/messages            (anthropic-version: 2023-06-01)
//   gemini    POST <base>/v1beta/models/<model>:generateContent   (JSON)
//             POST <base>/v1beta/models/<model>:streamGenerateContent (SSE)
// The model is in the URL path on the Gemini wire; the body `model` is ignored there.
// Auth is per provider, never per wire: bearer / x-api-key / x-goog-api-key / none.

import { authHeaders, isRecord, joinBase, type Json, type ProviderDirectoryEntry, type Wire } from './providers.ts';

/** One image, as the client sent it. `data` is raw base64 for a data: URL, or the URL itself. */
export interface ImageData {
  mimeType: string;
  data: string;
  isUrl: boolean;
}

export interface NormTool {
  name: string;
  description: string;
  parameters: Json;
}

/** A client request with the wire stripped off. Single-turn only: the client sends a
 *  system prompt (sometimes), one user message (text, maybe an image), tool definitions,
 *  a stop list, a max_tokens cap and maybe JSON mode. No conversation history ever arrives. */
export interface NormalizedRequest {
  system: string | null;
  userText: string;
  image: ImageData | null;
  tools: NormTool[];
  stop: string[];
  maxTokens: number | null;
  jsonMode: boolean;
  temperature: number | null;
}

export interface UpstreamAnswer {
  text: string;
  inputTokens: number;
  outputTokens: number;
}

export interface UpstreamFailure {
  status: number;
  message: string;
  code: string;
}

export type UpstreamResult = { ok: true; answer: UpstreamAnswer } | { ok: false; failure: UpstreamFailure };

/** How long one upstream call may take before it is abandoned (the client waits 30 s). */
const UPSTREAM_TIMEOUT_MS = 5000;

/** Split a `data:<mime>;base64,<b64>` URL; null when it is not one. */
function parseDataUrl(url: string): { mimeType: string; data: string } | null {
  const m = /^data:([^;,]+);base64,(.*)$/s.exec(url);
  if (m === null) return null;
  return { mimeType: m[1], data: m[2] };
}

/** Text of a string or of a block array's text blocks, joined with "\n". */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('\n');
}

function stringOf(v: unknown): string {
  return typeof v === 'string' ? v : '';
}

function numberOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

/** The image of an OpenAI `image_url` part, or null. */
function openAIImage(part: Json): ImageData | null {
  if (part.type !== 'image_url' || !isRecord(part.image_url)) return null;
  const url = stringOf(part.image_url.url);
  if (url === '') return null;
  const data = parseDataUrl(url);
  if (data !== null) return { mimeType: data.mimeType, data: data.data, isUrl: false };
  return { mimeType: '', data: url, isUrl: true };
}

/** The image of an Anthropic `image` block, or null. */
function anthropicImage(block: Json): ImageData | null {
  if (block.type !== 'image' || !isRecord(block.source)) return null;
  const source = block.source;
  if (source.type === 'base64' && typeof source.data === 'string') {
    return { mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data, isUrl: false };
  }
  if (source.type === 'url' && typeof source.url === 'string') {
    return { mimeType: '', data: source.url, isUrl: true };
  }
  return null;
}

/** Extract a wire-neutral request from an OpenAI Chat Completions body. */
export function extractOpenAI(body: Json): NormalizedRequest {
  let system: string | null = null;
  let userText = '';
  let image: ImageData | null = null;
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      system = system === null ? text : `${system}\n${text}`;
      continue;
    }
    if (raw.role !== 'user') continue;
    const content = raw.content;
    if (typeof content === 'string') {
      userText = content;
      continue;
    }
    if (!Array.isArray(content)) continue;
    const texts: string[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'text' && typeof block.text === 'string') texts.push(block.text);
      const img = openAIImage(block);
      if (img !== null) image = img;
    }
    userText = texts.join('\n');
  }
  const tools: NormTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') continue;
      tools.push({
        name: raw.function.name,
        description: typeof raw.function.description === 'string' ? raw.function.description : '',
        parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
      });
    }
  }
  const stopRaw = body.stop;
  const stop: string[] = typeof stopRaw === 'string' ? [stopRaw] : Array.isArray(stopRaw) ? stopRaw.filter((s): s is string => typeof s === 'string') : [];
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    system: system === '' ? null : system,
    userText,
    image,
    tools,
    stop,
    maxTokens: numberOrNull(body.max_tokens) ?? numberOrNull(body.max_completion_tokens),
    jsonMode: format === 'json_object' || format === 'json_schema',
    temperature: numberOrNull(body.temperature),
  };
}

/** Extract a wire-neutral request from an Anthropic Messages body. */
export function extractAnthropic(body: Json): NormalizedRequest {
  const systemText = textOf(body.system);
  let userText = '';
  let image: ImageData | null = null;
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of messages) {
    if (!isRecord(raw) || raw.role !== 'user') continue;
    const content = raw.content;
    if (typeof content === 'string') {
      userText = content;
      continue;
    }
    if (!Array.isArray(content)) continue;
    const texts: string[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'text' && typeof block.text === 'string') texts.push(block.text);
      const img = anthropicImage(block);
      if (img !== null) image = img;
    }
    userText = texts.join('\n');
  }
  const tools: NormTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || typeof raw.name !== 'string') continue;
      tools.push({
        name: raw.name,
        description: typeof raw.description === 'string' ? raw.description : '',
        parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
      });
    }
  }
  const stop: string[] = Array.isArray(body.stop_sequences) ? body.stop_sequences.filter((s): s is string => typeof s === 'string') : [];
  return {
    system: systemText === '' ? null : systemText,
    userText,
    image,
    tools,
    stop,
    // The Anthropic wire has no JSON-mode field: jsonMode is always false here.
    maxTokens: numberOrNull(body.max_tokens),
    jsonMode: false,
    temperature: numberOrNull(body.temperature),
  };
}

/** The user message of an OpenAI upstream body. */
function openAIUserMessage(norm: NormalizedRequest): Json {
  if (norm.image === null) return { role: 'user', content: norm.userText };
  const parts: Json[] = [{ type: 'text', text: norm.userText }];
  parts.push({
    type: 'image_url',
    image_url: { url: norm.image.isUrl ? norm.image.data : `data:${norm.image.mimeType};base64,${norm.image.data}` },
  });
  return { role: 'user', content: parts };
}

/** The user message of an Anthropic upstream body. */
function anthropicUserMessage(norm: NormalizedRequest): Json {
  if (norm.image === null) return { role: 'user', content: norm.userText };
  const blocks: Json[] = [{ type: 'text', text: norm.userText }];
  if (norm.image.isUrl) blocks.push({ type: 'image', source: { type: 'url', url: norm.image.data } });
  else blocks.push({ type: 'image', source: { type: 'base64', media_type: norm.image.mimeType, data: norm.image.data } });
  return { role: 'user', content: blocks };
}

/** The user content of a Gemini upstream body. */
function geminiUserParts(norm: NormalizedRequest): Json[] {
  const parts: Json[] = [{ text: norm.userText }];
  if (norm.image !== null) {
    if (norm.image.isUrl) parts.push({ fileData: { fileUri: norm.image.data, mimeType: 'image/png' } });
    else parts.push({ inlineData: { mimeType: norm.image.mimeType, data: norm.image.data } });
  }
  return parts;
}

function openAITools(norm: NormalizedRequest): Json[] {
  return norm.tools.map((t) => ({ type: 'function', function: { name: t.name, description: t.description, parameters: t.parameters } }));
}

function anthropicTools(norm: NormalizedRequest): Json[] {
  return norm.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.parameters }));
}

function geminiTools(norm: NormalizedRequest): Json[] {
  return [{ functionDeclarations: norm.tools.map((t) => ({ name: t.name, description: t.description, parameters: t.parameters })) }];
}

/** Build one upstream request in the provider's own wire. Always non-streaming: the
 *  entry synthesizes the client's SSE itself (see lib/respond.ts), so the streamed-usage
 *  trap cannot bite. Returns the URL, headers and JSON body. */
export function buildUpstream(
  envUrl: string,
  provider: ProviderDirectoryEntry,
  modelId: string,
  norm: NormalizedRequest,
): { url: string; headers: Record<string, string>; body: Json } {
  const base = joinBase(envUrl, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (provider.wire === 'openai') {
    const body: Json = { model: modelId, messages: [] as Json[] };
    const messages = body.messages as Json[];
    if (norm.system !== null) messages.push({ role: 'system', content: norm.system });
    messages.push(openAIUserMessage(norm));
    if (norm.maxTokens !== null) body.max_tokens = norm.maxTokens;
    if (norm.temperature !== null) body.temperature = norm.temperature;
    if (norm.tools.length > 0) {
      body.tools = openAITools(norm);
      body.tool_choice = 'auto';
    }
    if (norm.jsonMode) body.response_format = { type: 'json_object' };
    if (norm.stop.length > 0) body.stop = norm.stop;
    body.stream = false;
    return { url: `${base}/v1/chat/completions`, headers, body };
  }
  if (provider.wire === 'anthropic') {
    headers['anthropic-version'] = '2023-06-01';
    const body: Json = { model: modelId, messages: [anthropicUserMessage(norm)] };
    if (norm.system !== null) body.system = norm.system;
    if (norm.maxTokens !== null) body.max_tokens = norm.maxTokens;
    if (norm.temperature !== null) body.temperature = norm.temperature;
    if (norm.tools.length > 0) body.tools = anthropicTools(norm);
    if (norm.stop.length > 0) body.stop_sequences = norm.stop;
    body.stream = false;
    return { url: `${base}/v1/messages`, headers, body };
  }
  // gemini: the model goes in the URL path; streaming is decided by the path suffix only.
  const body: Json = { contents: [{ role: 'user', parts: geminiUserParts(norm) }] };
  if (norm.system !== null) body.systemInstruction = { parts: [{ text: norm.system }] };
  if (norm.tools.length > 0) body.tools = geminiTools(norm);
  const config: Json = {};
  if (norm.maxTokens !== null) config.maxOutputTokens = norm.maxTokens;
  if (norm.stop.length > 0) config.stopSequences = norm.stop;
  if (norm.jsonMode) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) body.generationConfig = config;
  return { url: `${base}/v1beta/models/${encodeURIComponent(modelId)}:generateContent`, headers, body };
}

/** Pull the error message/code out of a provider's error envelope, per wire. */
function errorOf(wire: Wire, status: number, text: string): UpstreamFailure {
  let message = `HTTP ${status}`;
  let code = `http_${status}`;
  try {
    const parsed: unknown = JSON.parse(text);
    if (isRecord(parsed)) {
      if (wire === 'openai' && isRecord(parsed.error)) {
        if (typeof parsed.error.message === 'string') message = parsed.error.message;
        if (typeof parsed.error.code === 'string') code = parsed.error.code;
      } else if (wire === 'anthropic' && isRecord(parsed.error)) {
        if (typeof parsed.error.message === 'string') message = parsed.error.message;
        if (typeof parsed.error.type === 'string') code = parsed.error.type;
      } else if (wire === 'gemini' && isRecord(parsed.error)) {
        if (typeof parsed.error.message === 'string') message = parsed.error.message;
        if (typeof parsed.error.status === 'string') code = parsed.error.status;
      }
    }
  } catch {
    // Not JSON: keep the HTTP fallback.
  }
  return { status, message, code };
}

function intOrZero(v: unknown): number {
  return typeof v === 'number' && Number.isFinite(v) ? Math.trunc(v) : 0;
}

/** Parse a successful provider answer into {text, inputTokens, outputTokens}. */
function parseAnswer(wire: Wire, body: unknown): UpstreamAnswer | null {
  if (!isRecord(body)) return null;
  if (wire === 'openai') {
    const choices = Array.isArray(body.choices) ? body.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    const message = choice !== null && isRecord(choice.message) ? choice.message : null;
    const content = message !== null ? message.content : null;
    const text = typeof content === 'string' ? content : textOf(content);
    if (text === '') return null;
    const usage = isRecord(body.usage) ? body.usage : {};
    return { text, inputTokens: intOrZero(usage.prompt_tokens), outputTokens: intOrZero(usage.completion_tokens) };
  }
  if (wire === 'anthropic') {
    const content = Array.isArray(body.content) ? body.content : [];
    const text = content
      .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
      .join('\n');
    if (text === '') return null;
    const usage = isRecord(body.usage) ? body.usage : {};
    return { text, inputTokens: intOrZero(usage.input_tokens), outputTokens: intOrZero(usage.output_tokens) };
  }
  // gemini
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : null;
  const content = candidate !== null && isRecord(candidate.content) ? candidate.content : null;
  const parts = content !== null && Array.isArray(content.parts) ? content.parts : [];
  const text = parts
    .flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : []))
    .join('');
  if (text === '') return null;
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return { text, inputTokens: intOrZero(usage.promptTokenCount), outputTokens: intOrZero(usage.candidatesTokenCount) };
}

/** Call one provider once, non-streaming, and parse its answer. Network errors and
 *  timeouts come back as a 502 failure so the caller can fail over. */
export async function callUpstream(
  envUrl: string,
  provider: ProviderDirectoryEntry,
  modelId: string,
  norm: NormalizedRequest,
): Promise<UpstreamResult> {
  const { url, headers, body } = buildUpstream(envUrl, provider, modelId, norm);
  let res: Response;
  try {
    res = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
    });
  } catch (err) {
    return { ok: false, failure: { status: 502, message: `upstream unreachable: ${(err as Error).message}`, code: 'upstream_unreachable' } };
  }
  const text = await res.text();
  if (!res.ok) return { ok: false, failure: errorOf(provider.wire, res.status, text) };
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    return { ok: false, failure: { status: 502, message: 'provider answered non-JSON', code: 'bad_upstream' } };
  }
  const answer = parseAnswer(provider.wire, parsed);
  if (answer === null) return { ok: false, failure: { status: 502, message: 'provider answer had no text', code: 'bad_upstream' } };
  return { ok: true, answer };
}
