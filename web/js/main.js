// Orchestration: load the bundle, run a query through fly-in -> lines -> wall, expose window.__gallery.
import { Cloud } from './cloud.js';
import { Wall } from './wall.js';
import { Detail, Tip } from './detail.js';
import { Scope } from './scope.js';
import { Globe } from './globe.js';
import { tween, cancelExcept, ease } from './anim.js';

const FLY_MS = 900;
const LINE_MS = 300;
const LINE_STAGGER_MS = 60;
const DIM_TO = 0.22;

const $ = (id) => document.getElementById(id);
const input = $('q');
const errorEl = $('ask-error');
const weakBadge = $('weak-badge');

const state = { phase: 'idle', lastResult: null, pulsedIndex: null, run: 0, timings: {} };
let cloud, wall, detail, tip, scope, globe = null;
// the Moon set has one collection and one type: no scope to choose, no filters sent
const NO_SCOPE = { filters: () => ({}), label: () => 'All tiles', count: 0, isOpen: false, toggle() {} };

function setPhase(p) {
  state.phase = p;
  document.body.dataset.phase = p;
}

async function loadBundle() {
  const [itemsRes, pointsRes, metaRes] = await Promise.all([
    fetch('data/items.json'), fetch('data/points.bin'), fetch('data/meta.json'),
  ]);
  for (const r of [itemsRes, pointsRes, metaRes]) {
    if (!r.ok) throw new Error(`${r.url}: HTTP ${r.status}`);
  }
  const items = await itemsRes.json();
  const buf = await pointsRes.arrayBuffer();
  const meta = await metaRes.json();
  const n = items.length;
  if (buf.byteLength !== n * 15) throw new Error(`points.bin has ${buf.byteLength} bytes, expected ${n * 15}`);
  return { items, meta, xyz: new Float32Array(buf, 0, n * 3), rgb: new Uint8Array(buf, n * 12, n * 3) };
}

function showError(msg) {
  errorEl.textContent = msg;
  errorEl.hidden = false;
}

async function postSearch(query, filters) {
  let res;
  try {
    res = await fetch('search', {
      method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ query, filters }),
    });
  } catch (e) {
    throw new Error(`Could not reach the server (${e.message})`);
  }
  let body = null;
  try { body = await res.json(); } catch { /* non-JSON error body handled below */ }
  if (!res.ok) {
    const d = body && body.detail;
    const msg = typeof d === 'string' ? d : Array.isArray(d) ? d.map((x) => x.msg).join('; ') : `HTTP ${res.status}`;
    throw new Error(msg);
  }
  if (!body) throw new Error('The server returned an empty response');
  return body;
}

function lineOpacity(sim, lo, hi) {
  if (hi - lo < 1e-6) return 1;
  return 0.35 + 0.65 * (sim - lo) / (hi - lo);
}

async function runQuery(result, run) {
  const t0 = performance.now();
  state.timings = {};
  cancelExcept(run);
  cloud.clearResult();
  wall.hide();
  setPulsed(null);
  state.lastResult = result;
  weakBadge.hidden = !result.weak;
  detail.close();
  // on the Moon's globe: break it up into embedding space first
  if (globe && globe.active && !(await globe.morph(1, run))) return;
  syncGlobeToggle();
  await cloud.ensure3d();   // a query from the flat map first rounds it back into the cloud
  if (run !== state.run) return;
  cloud.mode = 'result';

  const { neighbors } = result;
  neighbors.forEach((n) => cloud.preload(n.index));
  const q = cloud.showQuery(result.position);
  const pts = [q, ...neighbors.map((n) => cloud.worldPos(n.index))];
  const d0 = cloud.uniforms.uDim.value;
  if (d0 < 1) tween(300, (e) => cloud.setDim(d0 + (1 - d0) * e), { run });

  setPhase('flying');
  if (!(await cloud.flyTo(pts, wall.panelWidth() / 2, FLY_MS, run))) return;
  state.timings.fly = performance.now() - t0;

  setPhase('lines');
  cloud.lastResult = neighbors;
  cloud.arrange(neighbors);
  const sims = neighbors.map((n) => n.similarity);
  const lo = Math.min(...sims), hi = Math.max(...sims);
  tween(LINE_STAGGER_MS * (neighbors.length - 1) + LINE_MS, (e) => cloud.setDim(1 - (1 - DIM_TO) * e),
    { run, easing: ease.inOutCubic });
  const grows = neighbors.map((n, i) => {
    const rec = cloud.addLine(n, lineOpacity(n.similarity, lo, hi), cloud.items[n.index].color);
    return cloud.growLine(rec, LINE_MS, i * LINE_STAGGER_MS, run);
  });
  const done = await Promise.all(grows);
  if (done.some((d) => !d) || run !== state.run) return;
  state.timings.lines = performance.now() - t0 - state.timings.fly;

  wall.render(result);
  wall.show();
  setPhase('wall');
  state.timings.wall = performance.now() - t0;
}

async function submit(e) {
  e.preventDefault();
  const query = input.value;
  if (!query.trim()) return;
  errorEl.hidden = true;
  input.disabled = true;
  let result;
  const tReq = performance.now();
  try {
    result = await postSearch(query, scope.filters());
  } catch (err) {
    showError(err.message);
    return;
  } finally {
    input.disabled = false;
    input.focus();
  }
  state.requestMs = performance.now() - tReq;
  const run = ++state.run;
  runQuery(result, run).catch((err) => {
    console.error(err);
    showError(`Display failed: ${err.message}`);
  });
}

function setPulsed(index) {
  state.pulsedIndex = index;
  cloud.pulse(index);
  wall.mark(index);
}

function closeWall() {
  const run = ++state.run;
  cancelExcept(run);
  wall.hide();
  weakBadge.hidden = true;
  setPulsed(null);
  setPhase('idle');
  cloud.reset(run);
}

const NAV_GLOBE = 'Drag to turn the Moon · scroll to zoom · click a tile';

function syncGlobeToggle() {
  if (!globe) return;
  const onGlobe = globe.t < 1;
  $('globe-toggle').textContent = onGlobe ? 'Into embedding space' : 'Back to the globe';
  $('globe-toggle').disabled = globe.busy;
  document.body.dataset.globe = onGlobe ? 'on' : 'off';
  if (onGlobe) $('nav-hint').textContent = NAV_GLOBE;
  else cloud.viewCb(cloud.view);
}

async function toggleGlobe() {
  if (!globe || globe.busy) return;
  const run = ++state.run;
  cancelExcept(run);
  const to = globe.t < 1 ? 1 : 0;
  if (to === 0) {   // leave any hung result and the flat map before gathering the tiles back
    wall.hide();
    weakBadge.hidden = true;
    setPulsed(null);
    detail.close();
    cloud.clearResult();
    cloud.setDim(1);
    await cloud.ensure3d();
    setPhase('idle');
  }
  const p = globe.morph(to, run);
  syncGlobeToggle();
  await p;
  syncGlobeToggle();
}

function benchOverlay() {
  const box = $('bench');
  box.hidden = false;
  const show = () => { box.textContent = `${cloud.medianFps()} fps · ${cloud.n} points · ${cloud.rendererString()}`; };
  show();
  setInterval(show, 500);
}

async function main() {
  let bundle;
  try {
    bundle = await loadBundle();
  } catch (err) {
    $('loading').textContent = `Could not load the collection: ${err.message}`;
    $('loading').classList.add('is-error');
    throw err;
  }
  const { items, meta, xyz, rgb } = bundle;
  cloud = new Cloud($('stage'), { xyz, rgb, items });
  wall = new Wall({
    items,
    onHover: (i) => setPulsed(i),
    onOpenPage: () => { wall.openPage(); setPhase('page'); },
    onBack: () => { wall.closePage(); setPhase('wall'); input.focus(); },
    onClose: closeWall,
  });
  cloud.hoverCb = (i) => { if (state.phase === 'wall') setPulsed(i); };
  detail = new Detail(items, $('detail'), () => document.body.classList.remove('detail-open'));
  tip = new Tip(items, $('dot-tip'));
  if (meta.globe) {
    globe = new Globe(cloud, meta.globe);
    globe.showWhole();
    scope = NO_SCOPE;
    $('scope-btn').style.display = 'none';   // its CSS display would override [hidden]
    $('globe-toggle').hidden = false;
    $('globe-toggle').addEventListener('click', toggleGlobe);
  } else {
    scope = new Scope(items, { onChange: (flags) => cloud.setMuted(flags) });
  }
  cloud.clickCb = (i) => { tip.show(null); detail.open(i); document.body.classList.add('detail-open'); };
  cloud.tipCb = (i, x, y) => tip.show(i, x, y);
  const hint = $('nav-hint');
  cloud.viewCb = (v) => {
    document.body.dataset.view = v === '3d' ? '3d' : 'flat';
    if (globe && globe.t < 1) { hint.textContent = NAV_GLOBE; return; }
    hint.textContent = v === '3d'
      ? 'Drag to turn · right-drag to pan · scroll to zoom in on any spot'
      : 'Drag to pan · scroll to zoom · click a work · zoom out to return to the cloud';
  };

  const counts = items.reduce((a, it) => ((a[it.source] = (a[it.source] || 0) + 1), a), {});
  $('holdings').textContent = meta.globe
    ? `The whole Moon in ${items.length.toLocaleString()} tiles · LOLA / USGS via OpenPlanetaryMap`
    : `${items.length.toLocaleString()} works · ${[
      counts.aic && 'Art Institute of Chicago', counts.nasa && 'NASA', counts.wiki && 'Wikipedia',
    ].filter(Boolean).join(' · ')}`;
  syncGlobeToggle();
  const notes = [];
  if (meta.synthetic) notes.push('Preview with synthetic data: placeholder works and a keyword stand-in for EmbeddingGemma.');
  if (meta.n_pending) notes.push(`${meta.n_pending.toLocaleString()} summaries still pending.`);
  if (notes.length) { $('data-note').textContent = notes.join(' '); $('data-note').hidden = false; }

  $('ask').addEventListener('submit', submit);
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && scope.isOpen) { scope.toggle(false); return; }
    if (e.key === 'Escape' && detail.isOpen) { detail.close(); return; }
    if (e.key === 'Escape' && wall.pageOpen) { wall.closePage(); setPhase('wall'); input.focus(); }
  });

  window.__cloud = cloud;  // debug handle for ad hoc checks
  window.__gallery = {
    get ready() { return !!state.ready; },
    get pointCount() { return cloud.n; },
    get phase() { return state.phase; },
    get lastResult() { return state.lastResult; },
    get pulsedIndex() { return state.pulsedIndex; },
    get fps() { return cloud.medianFps(); },
    get lineCount() { return cloud.lines.length; },
    get billboardCount() { return cloud.billboards.size; },
    get dim() { return cloud.uniforms.uDim.value; },
    get timings() { return { ...state.timings, request: state.requestMs }; },
    get renderer() { return cloud.rendererString(); },
    get view() { return cloud.view; },
    get scope() { return { filters: scope.filters(), label: scope.label(), count: scope.count }; },
    get morph() { return cloud.morph; },
    get globe() { return globe ? { t: globe.t, busy: globe.busy } : null; },
    toggleGlobe,
    get mapThumbCount() { return cloud.mapSprites.size; },
    get mapThumbIndices() { return [...cloud.mapSprites.keys()]; },
    get pane() { const p = cloud.pane; return p ? { members: p.members.length, reanchors: cloud.reanchors || 0, relaxMs: cloud.relaxMs, spacing: cloud.minSpacingPx } : null; },
    get detailIndex() { return detail.index; },
    get cameraDist() { return cloud.camera.position.distanceTo(cloud.controls.target); },
    // screen position of an item as drawn (for click tests)
    screenOf(i) { return cloud.project(new cloud.camera.position.constructor(cloud.shown[i * 3], cloud.shown[i * 3 + 1], cloud.shown[i * 3 + 2])); },
    pick(x, y) { return cloud.pick(x, y); },
    get autoRotate() { return cloud.controls.autoRotate; },
    get camera() { return cloud.camera.position.toArray(); },
    get lineOpacities() { return cloud.lines.map((l) => l.opacity); },
    billboardRects() { return cloud.billboardRects(); },
    // screen positions of the query marker and each neighbour, for framing checks
    screen() {
      if (!state.lastResult) return null;
      return {
        query: cloud.project(cloud.queryPos),
        neighbors: state.lastResult.neighbors.map((n) => cloud.project(cloud.worldPos(n.index))),
        viewport: { w: window.innerWidth, h: window.innerHeight, wall: wall.panelWidth() },
      };
    },
  };
  setPhase('idle');
  if (new URLSearchParams(location.search).has('bench')) benchOverlay();
  await cloud.ready;
  $('loading').classList.add('is-done');
  input.focus();
  // "ready" once the loading veil has faded (700 ms), i.e. when the presenter can actually see the cloud
  setTimeout(() => { state.ready = true; }, 750);
}

main();
