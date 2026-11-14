// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// The entry reads the brief ($MAKEFUN_PROMPT_FILE), probes every provider's catalog
// route, registers the one catalog with the env, and serves the client's 20 requests,
// one at a time, each with the best free model that meets its requirements. Every
// provider call is non-streaming; the downstream JSON or SSE answer is synthesized
// from the provider's {text, usage, stop} (see lib/respond.ts), so the streamed-usage
// trap cannot bite: the client never sends stream_options, and a verbatim-forwarded
// OpenAI provider stream would carry no usage chunk.
//
// Routing (quality is 60% of the score): candidates are the free models meeting the
// request's requirements, ranked by the brief's rating_milli, ties broken by the
// provider's estimated remaining budget. On a refusal the entry fails over to the
// next candidate — never retries a dead provider, never serves a paid model, never
// makes a second successful call for one request. A fail-schedule refusal is not
// retried in place: failing over costs the same one call, and the quality ceiling
// (99) is structural — openrouter's 10–12 budget can never cover all 17 requests
// that want its 1475 model — so an extra call buys nothing the score can see, while
// the tie-break is fewest upstream calls. Only if every candidate refused does the
// entry retry the best one once, as a last resort: serving beats not serving.
//
// Determinism (final trials are replayed byte-for-byte): no wall clock, no unseeded
// randomness, strictly sequential upstream calls, fixed id/created values.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  capabilitiesOf,
  isRecord,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type ProviderDirectoryEntry,
  type Requirement,
} from './lib/providers.ts';
import { anthropicError, anthropicModelList, openAIError, openAIModelList, type ServedModel } from './lib/wire.ts';
import {
  callUpstream,
  extractAnthropic,
  extractOpenAI,
  type NormalizedRequest,
  type UpstreamAnswer,
} from './lib/upstream.ts';
import { anthropicJson, anthropicStreamBody, openAIJson, openAIStreamBody } from './lib/respond.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** The most upstream attempts (calls + one retry each) spent on one request. */
const MAX_ATTEMPTS_PER_REQUEST = 16;
/** A user text longer than this carries the service log: the request needs a
 *  200k-context model (the manifest says so; this is the body-derived fallback). */
const LONG_TEXT_MIN_CONTEXT = 4000;
const LONG_TEXT_CONTEXT = 200000;

/** The sealed per-provider budget ranges' upper ends (fma-spec.md §1). A provider is
 *  never tried past this many calls: beyond it the budget is certainly spent. */
const MAX_BUDGET: Record<string, number> = {
  groq: 12,
  cerebras: 6,
  gemini: 8,
  zai: 5,
  openrouter: 12,
  sambanova: 5,
  pollinations: 14,
  cloudflare: 8,
  ovh: 6,
  kilol: 16,
};

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order. */
let manifestIndex = 0;
/** Upstream calls made per provider id, across the whole trial. */
const callsMade = new Map<string, number>();
/** Providers whose budget is spent or that are out: never tried again this trial. */
const dead = new Set<string>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, deduplicated by model id:
 *  the same id may be free on several providers (glm-4.7-flash, gemma-4-31b, ...),
 *  and GET /v1/models must list exactly the free model ids, once each. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  const seen = new Set<string>();
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  provider: ProviderDirectoryEntry;
  modelId: string;
  rating: number;
}

function maxBudget(providerId: string): number {
  return MAX_BUDGET[providerId] ?? 16;
}

function remainingBudget(providerId: string): number {
  return maxBudget(providerId) - (callsMade.get(providerId) ?? 0);
}

/**
 * The free models that can serve one request, best first: rating_milli descending,
 * then the provider's estimated remaining budget, then provider and model id (so the
 * order is a pure function of the brief and the calls made so far). An OpenAI-wire
 * JSON-mode request never goes to an anthropic-wire provider: that wire cannot carry
 * JSON mode, so the provider would echo json_mode false and the echo_json check
 * would fail. (Anthropic-wire client rows never carry a JSON-mode field, so they may
 * be served by any wire.)
 */
function candidatesFor(requires: Requirement, clientWire: 'openai' | 'anthropic'): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      if (requires.json === true && clientWire === 'openai' && probe.provider.wire === 'anthropic') continue;
      out.push({ provider: probe.provider, modelId: model.id, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort(
    (a, b) =>
      b.rating - a.rating ||
      remainingBudget(b.provider.id) - remainingBudget(a.provider.id) ||
      (a.provider.id < b.provider.id ? -1 : a.provider.id > b.provider.id ? 1 : 0) ||
      (a.modelId < b.modelId ? -1 : a.modelId > b.modelId ? 1 : 0),
  );
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order,
 *  unioned with what the body itself shows (tools, image, JSON mode, long text). */
function requirementsFor(norm: NormalizedRequest): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  const req: Requirement = {};
  const merge = (r: Requirement | undefined): void => {
    if (r === undefined) return;
    if (r.tools === true) req.tools = true;
    if (r.vision === true) req.vision = true;
    if (r.json === true) req.json = true;
    if (r.min_context !== undefined) req.min_context = Math.max(req.min_context ?? 0, r.min_context);
  };
  merge(entry?.requires);
  merge({
    tools: norm.tools.length > 0,
    vision: norm.image !== null,
    json: norm.jsonMode,
    ...(norm.userText.length > LONG_TEXT_MIN_CONTEXT ? { min_context: LONG_TEXT_CONTEXT } : {}),
  });
  return req;
}

/** A refusal message that means the provider is dead for the rest of the trial:
 *  its budget is spent, or its outage has begun. Anything else is transient. */
function isDeadRefusal(message: string): boolean {
  return message.includes('free-tier budget is spent') || message.includes(' is out');
}

interface Served {
  modelId: string;
  answer: UpstreamAnswer;
}

/**
 * Serve one normalized request: try the candidates best-first, failing over on any
 * refusal. A budget/outage refusal marks the provider dead; any other refusal just
 * moves on to the next candidate. No in-place retry: a fail-schedule refusal is
 * transient, but failing over costs the same one call and the quality ceiling (99)
 * is structural — openrouter's 10–12 budget can never cover all 17 requests that
 * want its 1475 model — so an extra call buys nothing the score can see, while the
 * tie-break is fewest upstream calls. If every candidate refused, retry the best
 * one once as a last resort: serving beats not serving.
 */
async function serve(norm: NormalizedRequest, requires: Requirement, clientWire: 'openai' | 'anthropic'): Promise<Served | null> {
  const candidates = candidatesFor(requires, clientWire);
  let attempts = 0;
  let firstCandidate: Candidate | null = null;
  for (const cand of candidates) {
    if (dead.has(cand.provider.id)) continue;
    if ((callsMade.get(cand.provider.id) ?? 0) >= maxBudget(cand.provider.id)) continue;
    if (attempts >= MAX_ATTEMPTS_PER_REQUEST) break;
    firstCandidate ??= cand;
    attempts += 1;
    const res = await callUpstream(ENV_URL, cand.provider, cand.modelId, norm);
    callsMade.set(cand.provider.id, (callsMade.get(cand.provider.id) ?? 0) + 1);
    if (res.ok) return { modelId: cand.modelId, answer: res.answer };
    log(`${cand.provider.id}/${cand.modelId} refused (${res.failure.status} ${res.failure.code}): ${res.failure.message}`);
    if (isDeadRefusal(res.failure.message)) dead.add(cand.provider.id);
    // Any other refusal: move on to the next candidate.
  }
  if (firstCandidate !== null && attempts < MAX_ATTEMPTS_PER_REQUEST && !dead.has(firstCandidate.provider.id)) {
    const retry = await callUpstream(ENV_URL, firstCandidate.provider, firstCandidate.modelId, norm);
    callsMade.set(firstCandidate.provider.id, (callsMade.get(firstCandidate.provider.id) ?? 0) + 1);
    if (retry.ok) return { modelId: firstCandidate.modelId, answer: retry.answer };
    log(`${firstCandidate.provider.id}/${firstCandidate.modelId} last-resort retry refused (${retry.failure.status} ${retry.failure.code}): ${retry.failure.message}`);
  }
  return null;
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendSse(res: ServerResponse, body: string): void {
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  res.end(body);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by any provider wire. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const norm = extractOpenAI(body);
  const served = await serve(norm, requirementsFor(norm), 'openai');
  if (served === null) return sendError(res, 'openai', 503, 'server_error', 'no free model could serve this request');
  if (body.stream === true) return sendSse(res, openAIStreamBody(served.modelId, served.answer));
  sendJson(res, 200, openAIJson(served.modelId, served.answer));
}

/** POST /v1/messages: an Anthropic-shaped client, answered by any provider wire. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const norm = extractAnthropic(body);
  const served = await serve(norm, requirementsFor(norm), 'anthropic');
  if (served === null) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model could serve this request');
  if (body.stream === true) return sendSse(res, anthropicStreamBody(served.modelId, served.answer));
  sendJson(res, 200, anthropicJson(served.modelId, served.answer));
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
