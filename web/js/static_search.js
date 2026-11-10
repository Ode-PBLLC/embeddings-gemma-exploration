// Search in the browser, for the static site (meta.static set by scripts/build_static.py).
// rank() is a port of gallery.search.Searcher.search: same filters, ranking, weak flag and query
// placement, same response shape as POST /search. tests/test_static_search.py holds the two together.
// Queries are embedded by EmbeddingGemma 2 in a worker (embed_worker.js); the suggested searches
// come with vectors precomputed by the server's own code, so they work before the model has loaded.

export const DIM = 768;
const MAX_QUERY_CHARS = 500;
const SOURCES = ['aic', 'nasa', 'wiki', 'moon'];
const MODALITIES = ['image', 'text', 'video', 'audio'];
const FIELDS = { modality: MODALITIES, source: SOURCES, artist: null, year: null };

export class QueryError extends Error {}

export function validateQuery(q) {
  if (typeof q !== 'string' || !q.trim()) throw new QueryError('query is empty');
  if (q.length > MAX_QUERY_CHARS) throw new QueryError(`query is longer than ${MAX_QUERY_CHARS} characters`);
  return q.trim();
}

export function validateFilters(filters) {
  const out = {};
  if (!filters) return out;
  for (const [field, values] of Object.entries(filters)) {
    if (!(field in FIELDS)) throw new QueryError(`unknown filter field '${field}'`);
    if (field === 'year') {
      if (!Array.isArray(values) || values.length !== 2 || !values.every((v) => typeof v === 'number' && Number.isFinite(v))) {
        throw new QueryError("filter 'year' must be [from, to] (numbers; negative = BCE)");
      }
      if (values[0] > values[1]) throw new QueryError(`filter 'year': from ${values[0]} is after to ${values[1]}`);
      out.year = [values[0], values[1]];
      continue;
    }
    if (!Array.isArray(values) || !values.every((v) => typeof v === 'string')) {
      throw new QueryError(`filter '${field}' must be a list of strings`);
    }
    if (!values.length) continue;
    const allowed = FIELDS[field];
    if (allowed === null) { out[field] = [...new Set(values)].sort(); continue; }
    const bad = values.filter((v) => !allowed.includes(v));
    if (bad.length) throw new QueryError(`unknown ${field} value(s) ${bad.join(', ')}`);
    out[field] = [...new Set(values)].sort((a, b) => allowed.indexOf(a) - allowed.indexOf(b));
  }
  return out;
}

// items.json + points.bin + vectors.bin -> what rank() needs
export function makeIndex(items, xyz, vectors, config) {
  const n = items.length;
  if (vectors.length !== n * DIM) throw new Error(`vectors.bin holds ${vectors.length / DIM} vectors for ${n} items`);
  if (xyz.length !== n * 3) throw new Error(`points.bin holds ${xyz.length / 3} points for ${n} items`);
  if (n < config.k) throw new Error(`only ${n} items, fewer than k=${config.k}`);
  return {
    n, xyz, vectors, config,
    ids: items.map((it) => it.id),
    sources: items.map((it) => it.source),
    modalities: items.map((it) => it.modality),
    artists: items.map((it) => it.artist || ''),
    years: items.map((it) => it.years || null),
  };
}

function eligible(ix, f) {
  const mask = new Uint8Array(ix.n).fill(1);
  if (f.artist) {
    const known = new Set(ix.artists);
    const unknown = f.artist.filter((a) => !known.has(a));
    if (unknown.length) throw new QueryError(`unknown artist(s) ${unknown.join(', ')}`);
  }
  for (let i = 0; i < ix.n; i++) {
    if (f.source && !f.source.includes(ix.sources[i])) mask[i] = 0;
    else if (f.modality && !f.modality.includes(ix.modalities[i])) mask[i] = 0;
    else if (f.artist && !f.artist.includes(ix.artists[i])) mask[i] = 0;
    else if (f.year) {   // undated works are not eligible
      const y = ix.years[i];
      if (!y || y[0] == null || y[1] == null || !(y[1] >= f.year[0] && y[0] <= f.year[1])) mask[i] = 0;
    }
  }
  return mask;
}

const bySim = (sims) => (a, b) => (sims[b] - sims[a]) || (a - b);

function pick(ix, sims, mask) {
  const { k, mode } = ix.config;
  const cand = [];
  for (let i = 0; i < ix.n; i++) if (mask[i]) cand.push(i);
  let top;
  if (mode === 'raw') {
    top = cand.sort(bySim(sims)).slice(0, k);
  } else {   // quota: k / (sources present) from each source
    const present = SOURCES.filter((s) => cand.some((i) => ix.sources[i] === s));
    const per = Math.floor(k / present.length);
    top = present.flatMap((s) => cand.filter((i) => ix.sources[i] === s).sort(bySim(sims)).slice(0, per));
  }
  return top.sort(bySim(sims));
}

// qv: unit Float32Array(768). Returns the POST /search response.
export function rank(ix, query, qv, filters) {
  const q = validateQuery(query);
  const f = validateFilters(filters);
  const mask = eligible(ix, f);
  let nEligible = 0;
  for (let i = 0; i < ix.n; i++) nEligible += mask[i];
  if (!nEligible) throw new QueryError('no works match these filters');
  if (qv.length !== DIM) throw new Error(`embedder returned a vector of length ${qv.length}`);
  let norm = 0;
  for (let j = 0; j < DIM; j++) norm += qv[j] * qv[j];
  norm = Math.sqrt(norm);
  if (!Number.isFinite(norm) || norm === 0) throw new Error('embedder returned a bad vector');
  const sims = new Float32Array(ix.n);
  let best = -Infinity;
  for (let i = 0; i < ix.n; i++) {
    let s = 0;
    const o = i * DIM;
    for (let j = 0; j < DIM; j++) s += ix.vectors[o + j] * qv[j];
    sims[i] = s / norm;
    if (mask[i] && sims[i] > best) best = sims[i];
  }
  const top = pick(ix, sims, mask);
  const pos = [0, 0, 0];
  let wsum = 0;
  for (const i of top) {
    const w = Math.max(sims[i], 0);
    wsum += w;
    for (let c = 0; c < 3; c++) pos[c] += w * ix.xyz[i * 3 + c];
  }
  if (wsum > 0) for (let c = 0; c < 3; c++) pos[c] /= wsum;
  else for (let c = 0; c < 3; c++) pos[c] = top.reduce((a, i) => a + ix.xyz[i * 3 + c], 0) / top.length;
  return {
    query: q,
    position: pos,
    neighbors: top.map((i, r) => ({ index: i, id: ix.ids[i], similarity: sims[i], rank: r + 1 })),
    weak: best < ix.config.weak_threshold,
    mode: ix.config.mode,
    filters: f,
    eligible: nEligible,
  };
}

export function decodeVector(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

const MB = (b) => Math.round(b / 1e6);

// The search the page calls instead of POST /search. onStatus(s) gets the model's progress:
// {state: 'downloading'|'preparing'|'ready'|'failed', loaded, total, device, cached, message}.
export class StaticSearch {
  static async create(meta, items, xyz, onStatus) {
    const [vr, dr] = await Promise.all([fetch('data/vectors.bin'), fetch('data/demo_queries.json')]);
    for (const r of [vr, dr]) if (!r.ok) throw new Error(`${r.url}: HTTP ${r.status}`);
    const vectors = new Float32Array(await vr.arrayBuffer());
    const demos = await dr.json();
    return new StaticSearch(makeIndex(items, xyz, vectors, meta.static.retrieval), demos, meta.static.model, onStatus);
  }

  constructor(index, demos, model, onStatus) {
    this.index = index;
    this.demos = new Map(demos.map((d) => [d.q, decodeVector(d.v)]));
    this.suggestions = demos.map((d) => d.q);
    this.model = model;
    this.onStatus = onStatus;
    this.status = { state: 'downloading', loaded: 0, total: 0 };
    this.pending = new Map();
    this.nextId = 0;
    this.worker = new Worker(new URL('./embed_worker.js', import.meta.url), { type: 'module' });
    this.worker.onmessage = (e) => this.onMessage(e.data);
    this.worker.onerror = (e) => this.setStatus({ state: 'failed', message: e.message || 'the model worker failed to start' });
    this.worker.postMessage({ type: 'load', model });
  }

  setStatus(s) {
    this.status = { ...this.status, ...s };
    if (s.state === 'failed') {
      console.error('search model failed:', s.message);
      for (const { reject } of this.pending.values()) reject(new Error(`The search model failed to load: ${s.message}`));
      this.pending.clear();
    }
    this.onStatus(this.status);
  }

  onMessage(m) {
    if (m.type === 'status') { this.setStatus(m.status); return; }
    const p = this.pending.get(m.id);
    if (!p) return;
    this.pending.delete(m.id);
    if (m.type === 'vector') p.resolve(new Float32Array(m.vector));
    else p.reject(new Error(`Embedding failed: ${m.message}`));
  }

  embed(text) {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: 'embed', id, text: this.model.query_prompt + text });
    });
  }

  async search(query, filters) {
    const q = validateQuery(query);
    let qv = this.demos.get(q);
    if (!qv) {
      const s = this.status;
      if (s.state === 'failed') throw new Error(`The search model failed to load: ${s.message}`);
      if (s.state !== 'ready') {
        const where = s.state === 'downloading' && s.total ? ` (${MB(s.loaded)} of ${MB(s.total)} MB)` : '';
        throw new Error(`The search model is still loading${where}. Try a suggested search, or ask again in a moment.`);
      }
      qv = await this.embed(q);
    }
    return rank(this.index, q, qv, filters);
  }
}
