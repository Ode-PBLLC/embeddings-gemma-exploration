// The labelling page: the gallery cloud as a tool for building a label tree.
// Draw around works -> the server checks the selection against the full vectors -> name it -> sub-cluster,
// split by words, or review suggestions. The Cloud (web/js/cloud.js) is used as it is; this file only
// reads its positions and camera, rewrites its dot colours, and draws marks on a canvas over it.
import * as THREE from 'three';
import { Cloud } from '/js/cloud.js';
import { Detail, Tip } from '/js/detail.js';
import { tween, cancelExcept, ease } from '/js/anim.js';

const PALETTE = ['#b5482f', '#2f6f8f', '#c98a2b', '#4d7c4a', '#7a4f8f', '#c2577a', '#3d8f86', '#8a6a3b',
  '#56609e', '#9aa13a', '#d07a4a', '#5b8fb0'];
const GHOST = [222, 218, 209];      // where unlabelled dots fade to: a warm grey just off the wall
const OFF = [176, 171, 162];        // in the working set, but switched off or unsure
const FLY_MS = 900;
const SAVE_MS = 600;
const MIN_LASSO_AREA = 150;   // px²: anything smaller was a click, not a selection
const LASSO_STEP_PX = 3;      // record a point of the outline every few pixels of pointer travel

const $ = (id) => document.getElementById(id);
const fmt = (n) => n.toLocaleString();
const plural = (n, one, many = one + 's') => `${fmt(n)} ${n === 1 ? one : many}`;
const cap = (s) => (s ? s[0].toUpperCase() + s.slice(1) : s);
const stamp = () => new Date().toISOString().replace(/\.\d+Z$/, 'Z');
const rgbOf = (hex) => { const v = parseInt(hex.slice(1), 16); return [(v >> 16) & 255, (v >> 8) & 255, v & 255]; };

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}

function button(label, cls, onClick) {
  const b = el('button', `btn ${cls || ''}`.trim(), label);
  b.type = 'button';
  b.addEventListener('click', onClick);
  return b;
}

let items, n, cloud, detail, tip, baseRgb, colorAttr;
const S = { nodes: [], labels: [], pending: [], rejected: [] };   // what is saved
let ix = null;                // indexes over S, rebuilt by reindex()
let focus = null;             // the label being worked in (null = all works)
let tray = null;              // the open step on the right: {kind, ...}
let preview = null;           // {groupOf: Int16Array, rgb: [[r,g,b]]} while a tray recolours the dots
let hover = null;             // {items, color}: works ringed because the pointer is on their card
let lasso = null;             // [[x, y], ...]: the outline being drawn, in screen pixels
let armed = false;            // the Select tool is on
let labelRgb = [];            // per item: the label colour its dot is painted with, or null
let flyRun = 100;
let saveTimer = null;
let ready = false;

// ---------- server ----------

async function api(path, body, method = 'POST') {
  let res;
  try {
    res = await fetch(`label/api/${path}`, {
      method, headers: { 'content-type': 'application/json' }, body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch (e) {
    throw new Error(`Could not reach the server (${e.message})`);
  }
  let data = null;
  try { data = await res.json(); } catch { /* a non-JSON error body is reported by status below */ }
  if (!res.ok) {
    const d = data && data.detail;
    const err = new Error(typeof d === 'string' ? d : Array.isArray(d) ? d.map((x) => x.msg).join('; ') : `HTTP ${res.status}`);
    err.status = res.status;
    throw err;
  }
  return data;
}

function showError(msg) {
  const e = $('page-error');
  e.textContent = msg;
  e.hidden = false;
  clearTimeout(showError.t);
  showError.t = setTimeout(() => { e.hidden = true; }, 9000);
}

const guard = (fn) => async (...a) => {
  try { return await fn(...a); } catch (err) { console.error(err); showError(err.message); return undefined; }
};

// ---------- the label model ----------

function reindex() {
  ix = { byId: new Map(), kids: new Map([[null, []]]), own: new Map(), pending: new Map(), members: new Map() };
  for (const nd of S.nodes) { ix.byId.set(nd.id, nd); ix.kids.set(nd.id, []); ix.own.set(nd.id, new Map()); ix.pending.set(nd.id, []); }
  for (const nd of S.nodes) ix.kids.get(nd.parent ?? null).push(nd);
  for (const r of S.labels) ix.own.get(r.node).set(r.i, r);
  for (const r of S.pending) ix.pending.get(r.node).push(r);
}

const kids = (id) => ix.kids.get(id ?? null) || [];

// A label's works: its own and those of every label under it (a "Gold" coin is one of the "Coins").
function members(id) {
  if (id == null) return null;
  let set = ix.members.get(id);
  if (!set) {
    set = new Set(ix.own.get(id).keys());
    for (const k of kids(id)) for (const i of members(k.id)) set.add(i);
    ix.members.set(id, set);
  }
  return set;
}

function descendants(id) {
  const out = [id];
  for (const k of kids(id)) out.push(...descendants(k.id));
  return out;
}

function pathOf(id) {
  const parts = [];
  for (let nd = ix.byId.get(id); nd; nd = ix.byId.get(nd.parent)) parts.unshift(nd.name);
  return parts.join(' › ');
}

function nextColor(parent) {
  const used = new Set(kids(parent).map((k) => k.color));
  if (parent != null) used.add(ix.byId.get(parent).color);
  return PALETTE.find((c) => !used.has(c)) || PALETTE[kids(parent).length % PALETTE.length];
}

// The colours the next k labels under a parent will get, so a preview shows what will be kept.
function freshColors(parent, k) {
  const used = new Set(kids(parent).map((x) => x.color));
  if (parent != null) used.add(ix.byId.get(parent).color);
  const free = PALETTE.filter((c) => !used.has(c));
  return Array.from({ length: k }, (_, g) => (g < free.length ? free[g] : PALETTE[g % PALETTE.length]));
}

function addNode(name, parent, via, color) {
  const nd = { id: `n${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`, name, parent: parent ?? null,
    color: color || nextColor(parent), at: stamp(), via };
  S.nodes.push(nd);
  reindex();
  return nd;
}

// Confirmed labels. `meta` says where each came from: {source, method, review, confidence?, margin?}.
function addLabels(node, list, meta) {
  const own = ix.own.get(node);
  const at = stamp();
  const added = new Set();
  for (const i of list) {
    if (own.has(i) || added.has(i)) continue;
    added.add(i);
    S.labels.push({ i, node, ...meta, at });
  }
  S.pending = S.pending.filter((r) => !(r.node === node && added.has(r.i)));
  reindex();
  return added.size;
}

function removeLabel(i, node) {
  const gone = new Set(descendants(node));
  S.labels = S.labels.filter((r) => !(r.i === i && gone.has(r.node)));
}

function deleteNode(id) {
  const gone = new Set(descendants(id));
  for (const key of ['labels', 'pending', 'rejected']) S[key] = S[key].filter((r) => !gone.has(r.node));
  S.nodes = S.nodes.filter((nd) => !gone.has(nd.id));
}

// After any change to S: rebuild the indexes, repaint, redraw the panels, save.
function commit() {
  reindex();
  if (focus != null && !ix.byId.has(focus)) focus = null;
  paint();
  renderTree();
  renderActions();
  if (detail.isOpen) renderDetailLabels(detail.index);
  clearTimeout(saveTimer);
  saveTimer = setTimeout(save, SAVE_MS);
}

async function save() {
  const s = $('save-status');
  try {
    const r = await api('state', S, 'PUT');
    s.classList.remove('is-error');
    s.textContent = `Saved ${new Date(r.saved_at).toLocaleTimeString()} · ${plural(r.labels, 'label')}, ${fmt(r.pending)} waiting`;
    $('save-path').textContent = r.path.split('/').slice(-3).join('/');
  } catch (err) {
    s.classList.add('is-error');
    s.textContent = `Not saved: ${err.message}`;
  }
}

// ---------- the dots ----------

// Colour rule: a dot takes the colour of the label it falls in among the children of the label in
// focus (the top-level labels when nothing is in focus). Everything else fades toward the wall.
function paint() {
  const arr = colorAttr.array;
  const sets = kids(focus).map((k) => [members(k.id), rgbOf(k.color)]);
  const mine = focus != null ? members(focus) : null;
  const mineRgb = focus != null ? rgbOf(ix.byId.get(focus).color) : null;
  const fade = preview ? 0.82 : focus != null ? 0.8 : S.nodes.length ? 0.5 : 0;
  labelRgb = new Array(n).fill(null);
  for (let i = 0; i < n; i++) {
    let c = null;
    if (preview) {
      const g = preview.groupOf[i];
      c = g >= 0 ? preview.rgb[g] : g === -2 ? OFF : null;
    } else {
      for (const [set, col] of sets) if (set.has(i)) { c = col; break; }
      if (!c && mine && mine.has(i)) c = mineRgb;
    }
    labelRgb[i] = c;
    for (let a = 0; a < 3; a++) arr[i * 3 + a] = c ? c[a] : Math.round(baseRgb[i * 3 + a] + (GHOST[a] - baseRgb[i * 3 + a]) * fade);
  }
  colorAttr.needsUpdate = true;
}

const V = new THREE.Vector3();
function screenOf(i) {
  V.set(cloud.shown[i * 3], cloud.shown[i * 3 + 1], cloud.shown[i * 3 + 2]);
  return cloud.project(V);
}

// Inside a closed outline (even-odd rule, so a figure-eight selects both loops).
function inside(poly, x, y) {
  let hit = false;
  for (let a = 0, b = poly.length - 1; a < poly.length; b = a++) {
    const [xa, ya] = poly[a], [xb, yb] = poly[b];
    if ((ya > y) !== (yb > y) && x < xa + ((y - ya) * (xb - xa)) / (yb - ya)) hit = !hit;
  }
  return hit;
}

function polyArea(poly) {
  let s = 0;
  for (let a = 0, b = poly.length - 1; a < poly.length; b = a++) s += (poly[b][0] + poly[a][0]) * (poly[b][1] - poly[a][1]);
  return Math.abs(s) / 2;
}

// The works an outline drawn on the screen catches: every dot (or picture centre) inside it. On the
// thumbnail pane only the pane's own works count (the rest of the cloud is dimmed behind it).
function caughtBy(poly) {
  const flat = cloud.view !== '3d';
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const [x, y] of poly) { x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); }
  const out = [];
  for (let i = 0; i < n; i++) {
    if (cloud.away[i] > 1.5 || (flat && cloud.away[i] > 0.5)) continue;
    const p = screenOf(i);
    if (p.z < -1 || p.z > 1 || p.x < x0 || p.x > x1 || p.y < y0 || p.y > y1) continue;
    if (inside(poly, p.x, p.y)) out.push(i);
  }
  return out;
}

// Rings and the selection outline, drawn over the cloud every frame.
function drawMarks() {
  requestAnimationFrame(drawMarks);
  const cv = $('marks');
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const W = window.innerWidth, H = window.innerHeight;
  if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
  const g = cv.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, W, H);
  if (!ready) return;
  const tanV = Math.tan(THREE.MathUtils.degToRad(cloud.camera.fov) / 2);
  const ring = (i, color, radius, { dash = null, width = 1.6 } = {}) => {
    if (cloud.away[i] > 1.5) return;
    const p = screenOf(i);
    if (p.z > 1 || p.z < -1 || p.x < -60 || p.x > W + 60 || p.y < -60 || p.y > H + 60) return;
    g.setLineDash(dash || []);
    g.lineWidth = width;
    g.strokeStyle = color;
    const rec = cloud.mapSprites.get(i);
    g.beginPath();
    if (rec && rec.sprite.material.map) {   // shown as a picture: outline the picture
      const w = rec.sprite.scale.x * H / (2 * tanV) + 6, h = rec.sprite.scale.y * H / (2 * tanV) + 6;
      g.rect(p.x - w / 2, p.y - h / 2, w, h);
    } else {
      g.arc(p.x, p.y, radius, 0, Math.PI * 2);
    }
    g.stroke();
  };
  // suggestions waiting on the label in focus (or on every label): dashed, in the label's colour
  const under = focus != null ? new Set(descendants(focus)) : null;
  for (const r of S.pending) {
    if (under && !under.has(r.node)) continue;
    ring(r.i, ix.byId.get(r.node).color, 7, { dash: [3, 3] });
  }
  if (tray && tray.kind === 'select' && tray.sel) {
    for (const grp of tray.sel.groups) for (const i of grp.items) ring(i, grp.on ? '#23201c' : 'rgba(35,32,28,.28)', 6);
    for (const i of tray.sel.strays.items) ring(i, tray.sel.strays.on ? '#23201c' : '#c98a2b', 6, { dash: [2, 3] });
  }
  if (hover) for (const i of hover.items) ring(i, hover.color || '#23201c', 9, { width: 2.2 });
  // on the pane the dots are pictures: show each picture's label as a small disc under it
  if (cloud.mapSprites.size) {
    for (const [i, rec] of cloud.mapSprites) {
      const c = labelRgb[i];
      if (!c) continue;
      const p = cloud.project(rec.sprite.position);
      const half = rec.sprite.scale.y * H / (4 * tanV);
      g.setLineDash([]);
      g.fillStyle = `rgb(${c[0]},${c[1]},${c[2]})`;
      g.strokeStyle = '#fdfcf9'; g.lineWidth = 1.5;
      g.beginPath(); g.arc(p.x, p.y + half - 2, 5, 0, Math.PI * 2); g.fill(); g.stroke();
    }
  }
  if (lasso && lasso.length > 1) {
    g.lineJoin = g.lineCap = 'round';
    g.beginPath();
    g.moveTo(lasso[0][0], lasso[0][1]);
    for (const [x, y] of lasso) g.lineTo(x, y);
    g.closePath();
    g.fillStyle = 'rgba(35,32,28,.06)';
    g.fill();
    // the drawn stroke solid; the gap that closes it back to the start dashed and faint
    g.setLineDash([]);
    g.lineWidth = 1.8;
    g.strokeStyle = '#23201c';
    g.beginPath();
    g.moveTo(lasso[0][0], lasso[0][1]);
    for (const [x, y] of lasso) g.lineTo(x, y);
    g.stroke();
    const [lx, ly] = lasso[lasso.length - 1];
    g.setLineDash([4, 5]);
    g.lineWidth = 1.2;
    g.strokeStyle = 'rgba(35,32,28,.45)';
    g.beginPath(); g.moveTo(lx, ly); g.lineTo(lasso[0][0], lasso[0][1]); g.stroke();
  }
}

// ---------- pictures that grow as you zoom into the pane ----------
// The gallery hangs a pane's pictures at a fixed size on screen, spaced so they just clear each other
// at PANE_FULL. Closer than that its layout only spreads further apart, so the pictures stay small
// and drift off-screen. Here they grow with the zoom instead, so the arrangement is simply magnified.
// (Done on this page's Cloud instance; the gallery's cloud.js is not changed.)
const PANE_FULL = 2.04;       // cloud.js: THUMB_D * 0.85, the distance its pane spacing is laid out for
const PANE_GROW_MAX = 4.5;    // about 330 px tall; the thumbnails are too coarse to enlarge further

function paneZoom() {
  if (cloud.view === '3d') return 1;
  const d = cloud.camera.position.distanceTo(cloud.controls.target);
  return Math.min(PANE_GROW_MAX, Math.max(1, PANE_FULL / d));
}

function installPaneZoom() {
  const stepMap = cloud.stepMap.bind(cloud);
  cloud.stepMap = () => {
    stepMap();
    const z = paneZoom();
    if (z > 1) for (const rec of cloud.mapSprites.values()) rec.sprite.scale.multiplyScalar(z);
  };
  // larger pictures reach the screen edge sooner: keep hanging the ones whose centre is just outside
  const hang = cloud.updateMapSprites.bind(cloud);
  cloud.updateMapSprites = (show, d) => {
    const z = paneZoom(), pane = cloud.pane;
    if (show.length && z > 1 && pane && pane.ruv) {
      const tanV = Math.tan(THREE.MathUtils.degToRad(cloud.camera.fov) / 2);
      const halfH = d * tanV + cloud.paneSpacing * z * 0.6, halfW = d * tanV * (window.innerWidth / window.innerHeight) + cloud.paneSpacing * z * 0.6;
      const { F, right, up } = cloud.paneBasis;
      const t = cloud.controls.target.clone().sub(F);
      const cu = t.dot(right), cv = t.dot(up);
      show = pane.members.filter((_, k) => Math.abs(pane.ruv[k * 2] - cu) < halfW && Math.abs(pane.ruv[k * 2 + 1] - cv) < halfH);
    }
    hang(show, d);
  };
}

// ---------- camera ----------

const sidePx = () => $('tree-panel').offsetWidth;
const shiftNow = () => ((tray ? sidePx() : 0) - sidePx()) / 2;   // centre the view in the space between the panels

function settleShift() {
  const s0 = cloud.shift, s1 = shiftNow();
  if (Math.abs(s0 - s1) < 1) return Promise.resolve(true);
  return tween(320, (e) => { cloud.shift = s0 + (s1 - s0) * e; cloud.applyShift(); }, { easing: ease.outCubic });
}

// A tray has opened over part of the view: put the works it is about in the space that is left.
// In the 3D cloud the camera flies to frame them (as the gallery does for a query's neighbours).
// On the thumbnail pane the camera only slides, so the pictures the person was looking at stay up.
async function bringIntoView(indices) {
  if (cloud.view === '3d') return flyTo(indices);
  await settleShift();
  if (cloud.view !== 'flat' || !indices || !indices.length) return;
  let x = 0, y = 0, k = 0;
  for (const i of indices) {
    if (cloud.away[i] > 0.5) continue;
    const p = screenOf(i);
    x += p.x; y += p.y; k++;
  }
  if (!k) return;
  const want = { x: sidePx() + (window.innerWidth - padsNow()) / 2, y: window.innerHeight / 2 };
  if (Math.hypot(x / k - want.x, y / k - want.y) < 60) return;
  const delta = cloud.planePoint(x / k, y / k).sub(cloud.planePoint(want.x, want.y));
  const p0 = cloud.camera.position.clone(), t0 = cloud.controls.target.clone();
  await tween(420, (e) => {
    cloud.camera.position.copy(p0).addScaledVector(delta, e);
    cloud.controls.target.copy(t0).addScaledVector(delta, e);
    cloud.camera.updateMatrixWorld();
  }, { easing: ease.outCubic });
  cloud.maybeReanchor();
  cloud.updateThumbs();
}

// Every work, framed in the space the panels leave free. Unlike the gallery's opening view this
// leaves no outliers off-screen: an isolated cluster (the coins) is exactly what one wants to select.
function homePoints() {
  const pts = [];
  for (let i = 0; i < n; i++) pts.push(cloud.worldPos(i));
  return pts;
}
const HOME_FIT = { fillX: 0.92, fillY: 0.8, minDist: 6 };
const SET_FIT = { minDist: 2.2 };     // close enough for a small set to hang as pictures, no closer
const padsNow = () => sidePx() + (tray ? sidePx() : 0);

// Fly to frame a set of works (a label's members, or everything), keeping the current viewing direction.
async function flyTo(indices) {
  const run = ++flyRun;
  cancelExcept(run);          // a flight still in the air stops where it is
  cloud.flying = false;
  await cloud.ensure3d();
  cloud.controls.enabled = true;
  if (run !== flyRun || (indices && !indices.length)) return;
  let pts = [];
  if (indices) {
    const step = Math.max(1, Math.floor(indices.length / 300));
    for (let k = 0; k < indices.length; k += step) pts.push(cloud.worldPos(indices[k]));
  } else {
    pts = homePoints();
  }
  const { target, position } = cloud.framing(pts, padsNow() / 2, indices ? SET_FIT : HOME_FIT);
  const p0 = cloud.camera.position.clone(), t0 = cloud.controls.target.clone(), s0 = cloud.shift, s1 = shiftNow();
  cloud.lastCursor = null;      // if the pane forms on arrival, form it around what was framed
  cloud.flying = true;
  cloud.controls.enabled = false;
  await tween(FLY_MS, (e) => {
    cloud.camera.position.lerpVectors(p0, position, e);
    cloud.controls.target.lerpVectors(t0, target, e);
    cloud.shift = s0 + (s1 - s0) * e;
    cloud.applyShift();
  }, { run, easing: ease.inOutCubic });
  if (run !== flyRun) return;
  cloud.settleControls();
  cloud.flying = false;
  cloud.controls.enabled = true;
  cloud.lastInteract = performance.now();
}

function setFocus(id, { fly = true } = {}) {
  focus = id ?? null;
  hover = null;
  paint();
  renderTree();
  renderActions();
  if (fly) flyTo(focus == null ? null : [...members(focus)]);
}

// ---------- small pieces shared by the trays ----------

function thumb(i) {
  const it = items[i];
  const b = el('button', 'thumb');
  b.type = 'button';
  b.title = it.title;
  if (it.thumb && it.modality !== 'text') {
    const img = el('img');
    img.loading = 'lazy'; img.alt = it.title; img.src = it.thumb;
    b.appendChild(img);
  } else {
    b.appendChild(el('span', null, it.title));
  }
  b.addEventListener('click', () => openDetail(i));
  b.addEventListener('pointerenter', () => { hover = { items: [i] }; });
  b.addEventListener('pointerleave', () => { hover = null; });
  return b;
}

function thumbs(list, max = 6) {
  const row = el('div', 'thumbs');
  for (const i of list.slice(0, max)) row.appendChild(thumb(i));
  return row;
}

function field(label, control, value) {
  const f = el('label', 'field');
  const s = el('span', null, label);
  if (value) s.appendChild(value);
  f.append(s, control);
  return f;
}

// A group of works as a card: tick box, colour, name (fixed text or an editable field), count, bar, pictures.
function groupCard({ list, color, name, editable, share, on = true, words, onToggle, onName, max = 6 }) {
  const card = el('div', 'card');
  card.classList.toggle('is-off', !on);
  if (color) card.style.setProperty('--c', color);
  const head = el('div', 'card-head');
  const box = el('input');
  box.type = 'checkbox'; box.checked = on; box.setAttribute('aria-label', 'Include this group');
  box.addEventListener('change', () => { card.classList.toggle('is-off', !box.checked); onToggle(box.checked); });
  const sw = el('span', 'tree-swatch');
  let nameEl;
  if (editable) {
    nameEl = el('input');
    nameEl.type = 'text'; nameEl.value = name; nameEl.spellcheck = false; nameEl.setAttribute('aria-label', 'Name of this group');
    nameEl.addEventListener('input', () => onName(nameEl.value));
  } else {
    nameEl = el('span', 'card-name', name);
  }
  head.append(box, sw, nameEl, el('span', 'card-count', fmt(list.length)));
  card.appendChild(head);
  if (share != null) {
    const bar = el('div', 'card-bar');
    const fill = el('i');
    fill.style.width = `${Math.max(1.5, share * 100)}%`;
    bar.appendChild(fill);
    card.appendChild(bar);
  }
  const wordsEl = el('p', 'card-words', words || '');
  wordsEl.hidden = !words;
  const pics = thumbs(list, max);
  card.append(wordsEl, pics);
  card.addEventListener('pointerenter', () => { hover = { items: list, color: color || undefined }; });
  card.addEventListener('pointerleave', () => { hover = null; });
  return { card, nameEl, wordsEl, pics, box };
}

// `show`: the works the step is about (an array, or null for all works), brought into the free space.
function openTray(next, kicker, show) {
  tray = next;
  $('tray').hidden = false;
  $('tray-kicker').textContent = kicker;
  document.body.classList.add('tray-open');
  if (show === undefined) settleShift(); else bringIntoView(show);
}

function closeTray() {
  if (!tray) return;
  tray = null;
  preview = null;
  hover = null;
  $('tray').hidden = true;
  $('tray-body').replaceChildren();
  document.body.classList.remove('tray-open');
  settleShift();
  paint();
}

function trayMessage(title, text, isError) {
  const p = el('p', 'tray-text', text);
  p.classList.toggle('is-error', !!isError);
  $('tray-body').replaceChildren(el('h2', 'tray-title', title), p);
}

// ---------- step 1: a drawn selection, checked against the full vectors ----------

async function openSelection(caught) {
  const inside = focus != null ? caught.filter((i) => members(focus).has(i)) : [];
  const outside = focus != null ? caught.filter((i) => !members(focus).has(i)) : caught;
  const name = focus != null ? ix.byId.get(focus).name : null;
  const options = [];
  if (inside.length) options.push({ key: 'inside', text: `A new label inside ${name} · ${plural(inside.length, 'work')}`, seed: inside, pool: [...members(focus)], parent: focus });
  if (focus != null && outside.length) options.push({ key: 'add', text: `Add to ${name} · ${plural(outside.length, 'work')} not in it yet`, seed: outside, pool: null, target: focus });
  options.push({ key: 'top', text: `A new top-level label · ${plural(caught.length, 'work')}`, seed: caught, pool: null, parent: null });
  const first = inside.length && inside.length >= outside.length ? 'inside' : options[0].key;
  openTray({ kind: 'select', options, choice: options.find((o) => o.key === first), sel: null, name: '' }, 'Selection', caught);
  await checkSelection();
}

async function checkSelection() {
  const t = tray, o = t.choice;
  t.sel = null;
  trayMessage(`${plural(o.seed.length, 'work')} selected`, 'Checking them against the full vectors…');
  let res, hint = null;
  try {
    res = await api('check', { seed: o.seed, pool: o.pool });
    // a hint for the name: words that set the main group apart from the rest of the label it is in,
    // or from a sample of everything when it is a top-level label
    const main = new Set(res.groups[0]);
    let rest = [];
    if (o.pool) rest = o.pool.filter((i) => !main.has(i));
    else for (let i = 0; i < n && rest.length < 300; i += Math.max(1, Math.floor(n / 300))) if (!main.has(i)) rest.push(i);
    hint = (await api('names', { groups: rest.length ? [res.groups[0], rest] : [res.groups[0]] })).groups[0].words;
  } catch (err) {
    if (tray === t) trayMessage('The check failed', err.message, true);
    return;
  }
  if (tray !== t) return;
  t.sel = { groups: res.groups.map((g, k) => ({ items: g, on: k === 0 })), strays: { items: res.strays, on: false }, hint };
  renderSelection();
}

function renderSelection() {
  const t = tray, sel = t.sel, o = t.choice;
  const body = $('tray-body');
  const total = o.seed.length;
  const parts = [];
  if (sel.groups.length === 1 && !sel.strays.items.length) {
    parts.push(total === 1 ? '' : 'They all sit together in the full vector space, not only in this view.');
  } else {
    parts.push(`In the full vector space these are not one group: ${plural(sel.groups[0].items.length, 'work')} sit together`
      + (sel.groups.length > 1 ? `, ${plural(sel.groups.length - 1, 'other group')} came along from elsewhere` : '')
      + (sel.strays.items.length ? `, and ${sel.strays.items.length} ${sel.strays.items.length === 1 ? 'stands' : 'stand'} alone` : '')
      + '. Tick what belongs.');
  }
  const kidsNote = el('p', 'tray-text', parts.join(' '));
  const nodes = [el('h2', 'tray-title', `${plural(total, 'work')} selected`), kidsNote];
  if (t.options.length > 1) {
    const pick = el('select');
    for (const opt of t.options) { const e = el('option', null, opt.text); e.value = opt.key; e.selected = opt === o; pick.appendChild(e); }
    pick.addEventListener('change', () => { t.choice = t.options.find((x) => x.key === pick.value); checkSelection(); });
    nodes.push(field('Make it', pick));
  }
  const go = button('', 'is-primary', guard(applySelection));
  const count = () => sel.groups.filter((g) => g.on).reduce((a, g) => a + g.items.length, 0) + (sel.strays.on ? sel.strays.items.length : 0);
  const sync = () => {
    const c = count();
    go.textContent = o.key === 'add' ? `Add ${plural(c, 'work')} to ${ix.byId.get(o.target).name}` : `Label ${plural(c, 'work')}`;
    go.disabled = c === 0 || (o.key !== 'add' && !t.name.trim());
  };
  sel.groups.forEach((grp, k) => {
    nodes.push(groupCard({
      list: grp.items, on: grp.on, share: grp.items.length / total,
      name: k === 0 ? (sel.groups.length > 1 || sel.strays.items.length ? 'Sit together' : 'All of them') : 'From elsewhere in the space',
      onToggle: (v) => { grp.on = v; sync(); },
    }).card);
  });
  if (sel.strays.items.length) {
    nodes.push(groupCard({
      list: sel.strays.items, on: sel.strays.on, share: sel.strays.items.length / total, name: 'On their own', max: 12,
      onToggle: (v) => { sel.strays.on = v; sync(); },
    }).card);
  }
  if (o.key !== 'add') {
    const input = el('input');
    input.type = 'text'; input.id = 'label-name'; input.value = t.name; input.spellcheck = false; input.autocomplete = 'off';
    input.placeholder = sel.hint && sel.hint.length ? `${cap(sel.hint[0])}…` : 'Name';
    const list = el('datalist');
    list.id = 'label-names';
    for (const k of kids(o.parent)) { const e = el('option'); e.value = k.name; list.appendChild(e); }
    input.setAttribute('list', 'label-names');
    input.addEventListener('input', () => { t.name = input.value; sync(); });
    input.addEventListener('keydown', (e) => { if (e.key === 'Enter' && !go.disabled) go.click(); });
    const f = field('Label', input);
    f.appendChild(list);
    nodes.push(f);
    if (sel.hint && sel.hint.length) nodes.push(el('p', 'card-words', `Words these share: ${sel.hint.join(', ')}`));
    setTimeout(() => input.focus(), 0);
  }
  const actions = el('div', 'tray-actions');
  actions.append(go, button('Cancel', 'is-quiet', closeTray));
  nodes.push(actions);
  body.replaceChildren(...nodes);
  sync();
}

async function applySelection() {
  const t = tray, sel = t.sel, o = t.choice;
  const list = [...sel.groups.filter((g) => g.on).flatMap((g) => g.items), ...(sel.strays.on ? sel.strays.items : [])];
  let node;
  if (o.key === 'add') {
    node = ix.byId.get(o.target);
  } else {
    const name = t.name.trim();
    node = kids(o.parent).find((k) => k.name.toLowerCase() === name.toLowerCase()) || addNode(name, o.parent, 'lasso');
  }
  addLabels(node.id, list, { source: 'person', method: 'lasso', review: 'by-hand' });
  closeTray();
  focus = node.id;
  commit();
  const found = await findMore(node.id);
  if (found) openReview(node.id);
}

// ---------- suggestions: neighbours vote ----------

// Ask for works that look like they belong to a label. Its siblings compete for each candidate (the
// server also leaves out works already in a sibling); works turned down before are not proposed again.
// Returns how many are waiting.
async function findMore(nodeId) {
  const nd = ix.byId.get(nodeId);
  const sibs = kids(nd.parent);
  const pool = nd.parent != null ? [...members(nd.parent)] : null;
  const labels = sibs.map((s) => ({
    node: s.id, members: [...members(s.id)], pool, group: nd.parent ?? 'root',
    rejected: S.rejected.filter((r) => r.node === s.id).map((r) => r.i),
  }));
  const { suggestions } = await api('suggest', { labels });
  if (!ix.byId.has(nodeId)) return 0;
  const had = new Set(S.pending.filter((r) => r.node === nodeId && r.method !== 'neighbour-vote').map((r) => r.i));
  S.pending = S.pending.filter((r) => !(r.node === nodeId && r.method === 'neighbour-vote'));
  const at = stamp();
  for (const s of suggestions) {
    if (s.node !== nodeId || had.has(s.item)) continue;
    S.pending.push({ i: s.item, node: nodeId, method: 'neighbour-vote', confidence: s.confidence, votes: s.votes, of: s.of, at });
  }
  commit();
  return ix.pending.get(nodeId).length;
}

function decide(row, yes, review = 'accepted') {
  S.pending = S.pending.filter((r) => r !== row);
  if (yes) {
    reindex();
    const meta = { source: 'suggestion', method: row.method, review };
    if (row.confidence != null) meta.confidence = row.confidence;
    if (row.margin != null) meta.margin = row.margin;
    addLabels(row.node, [row.i], meta);
  } else {
    S.rejected.push({ i: row.i, node: row.node, at: stamp() });
  }
}

function whyText(row, name) {
  if (row.method === 'neighbour-vote') return [`${Math.round(row.confidence * 100)}%`, ` · ${row.votes} of its ${row.of} nearest are ${name}`];
  if (row.method === 'text-split') return [`margin ${row.margin.toFixed(3)}`, ' · closest phrase, but not by much'];
  return ['', row.method];
}

function openReview(nodeId) {
  openTray({ kind: 'review', node: nodeId }, 'Suggestions', [...members(nodeId)]);
  renderReview();
}

function renderReview() {
  const t = tray, nd = ix.byId.get(t.node);
  if (!nd) { closeTray(); return; }
  const rows = [...ix.pending.get(nd.id)].sort((a, b) => (b.confidence ?? -1) - (a.confidence ?? -1) || (b.margin ?? 0) - (a.margin ?? 0));
  const body = $('tray-body');
  const more = button('Look again', '', guard(async () => { await findMore(nd.id); if (tray === t) renderReview(); }));
  if (!rows.length) {
    body.replaceChildren(el('h2', 'tray-title', `Nothing waiting for ${nd.name}`),
      el('p', 'tray-text', 'Each work you accept votes for its own neighbours, so looking again can turn up more.'));
    const a = el('div', 'tray-actions');
    a.append(more, button('Done', 'is-quiet', closeTray));
    body.appendChild(a);
    return;
  }
  const nodes = [el('h2', 'tray-title', `${plural(rows.length, 'work')} might be ${nd.name}`),
    el('p', 'tray-text', 'Proposed, not labelled. A work is proposed when enough of its nearest neighbours in the full vector space carry the label. Yes or no on each; a no is remembered.')];
  for (const row of rows) {
    const r = el('div', 'review-row');
    const mid = el('div');
    mid.appendChild(el('div', 'review-title', items[row.i].title));
    const why = el('div', 'review-why');
    const [strong, rest] = whyText(row, nd.name);
    why.append(el('b', null, strong), rest);
    mid.appendChild(why);
    const yn = el('div', 'yesno');
    const yes = el('button', 'yes', '✓'), no = el('button', 'no', '✕');
    yes.type = no.type = 'button';
    yes.setAttribute('aria-label', `Yes, ${items[row.i].title} is ${nd.name}`);
    no.setAttribute('aria-label', `No, ${items[row.i].title} is not ${nd.name}`);
    yes.addEventListener('click', () => { decide(row, true); hover = null; commit(); renderReview(); });
    no.addEventListener('click', () => { decide(row, false); hover = null; commit(); renderReview(); });
    yn.append(yes, no);
    r.append(thumb(row.i), mid, yn);
    r.addEventListener('pointerenter', () => { hover = { items: [row.i], color: nd.color }; });
    r.addEventListener('pointerleave', () => { hover = null; });
    nodes.push(r);
  }
  const sure = rows.filter((r) => (r.confidence ?? 0) >= 0.6);
  const actions = el('div', 'tray-actions');
  if (sure.length && sure.length < rows.length) {
    actions.appendChild(button(`Accept the ${sure.length} at 60% or more`, 'is-primary', () => {
      for (const row of sure) decide(row, true, 'accepted-in-bulk');
      commit(); renderReview();
    }));
  }
  actions.appendChild(button(`Accept all ${rows.length}`, sure.length && sure.length < rows.length ? '' : 'is-primary', () => {
    for (const row of rows) decide(row, true, 'accepted-in-bulk');
    commit(); renderReview();
  }));
  actions.append(more, button('Later', 'is-quiet', closeTray));
  nodes.push(actions);
  body.replaceChildren(...nodes);
}

// ---------- step 2: sub-cluster a label, with a granularity slider ----------

// Cut a Ward merge tree (scipy numbering) into k groups of leaf numbers, largest first.
function cut(merges, m, k) {
  const parent = new Int32Array(2 * m - 1);
  for (let i = 0; i < parent.length; i++) parent[i] = i;
  const find = (x) => { while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; };
  for (let t = 0; t < m - k; t++) { parent[find(merges[t][0])] = m + t; parent[find(merges[t][1])] = m + t; }
  const groups = new Map();
  for (let i = 0; i < m; i++) { const r = find(i); if (!groups.has(r)) groups.set(r, []); groups.get(r).push(i); }
  return [...groups.values()].sort((a, b) => b.length - a.length || a[0] - b[0]);
}

// Where the tree itself suggests cutting: the number of groups (2..8) just before the largest jump in merge height.
function naturalK(heights, m) {
  let best = 2, gap = -Infinity;
  for (let k = 2; k <= Math.min(8, m - 1); k++) {
    const d = heights[m - k] - heights[m - k - 1];   // cost of going from k groups down to k-1, over the step before
    if (d > gap) { gap = d; best = k; }
  }
  return best;
}

// The works a re-clustering sorts: a label's members not yet in one of its sub-labels (all of them if it has none).
function unsorted(nodeId) {
  if (nodeId == null) {
    const taken = new Set();
    for (const k of kids(null)) for (const i of members(k.id)) taken.add(i);
    return Array.from({ length: n }, (_, i) => i).filter((i) => !taken.has(i));
  }
  const taken = new Set();
  for (const k of kids(nodeId)) for (const i of members(k.id)) taken.add(i);
  return [...members(nodeId)].filter((i) => !taken.has(i));
}

async function openCluster(nodeId) {
  const target = unsorted(nodeId);
  const where = nodeId == null ? (kids(null).length ? 'the unlabelled works' : 'all works') : ix.byId.get(nodeId).name;
  const t = { kind: 'cluster', node: nodeId, where, names: new Map(), auto: new Map(), off: new Set() };
  openTray(t, 'Sub-cluster', nodeId == null && !kids(null).length ? null : target);
  if (target.length < 4) { trayMessage(`Nothing to sort in ${where}`, `Only ${plural(target.length, 'work')} here ${target.length === 1 ? 'is' : 'are'} not already in a sub-label.`); return; }
  trayMessage(`Sorting ${plural(target.length, 'work')}`, 'Building the merge tree from the full vectors (Ward)…');
  let res;
  try { res = await api('tree', { members: target }); } catch (err) { if (tray === t) trayMessage('Clustering failed', err.message, true); return; }
  if (tray !== t) return;
  Object.assign(t, { members: res.members, merges: res.merges, heights: res.heights, m: res.members.length });
  t.max = Math.min(24, Math.floor(t.m / 2));
  t.k = Math.min(t.max, naturalK(t.heights, t.m));
  renderCluster();
}

const groupKey = (leafs) => `${leafs[0]}:${leafs.length}`;

function renderCluster() {
  const t = tray;
  const groups = cut(t.merges, t.m, t.k).map((leafs) => ({ key: groupKey(leafs), list: leafs.map((l) => t.members[l]) }));
  t.groups = groups;
  const colors = t.colors = freshColors(t.node, groups.length);
  const repaint = () => {
    const groupOf = new Int16Array(n).fill(-1);
    groups.forEach((grp, g) => { for (const i of grp.list) groupOf[i] = t.off.has(grp.key) ? -2 : g; });
    preview = { groupOf, rgb: colors.map(rgbOf) };
    paint();
  };
  repaint();
  const slider = el('input');
  slider.type = 'range'; slider.min = 2; slider.max = t.max; slider.step = 1; slider.value = t.k; slider.id = 'granularity';
  const value = el('span', 'field-value', `${t.k} groups`);
  slider.addEventListener('input', () => { t.k = +slider.value; renderCluster(); $('granularity').focus(); });
  const keep = button('', 'is-primary', guard(applyCluster));
  const sync = () => {
    const c = groups.filter((g) => !t.off.has(g.key)).length;
    keep.textContent = `Keep ${plural(c, 'group')} as labels`;
    keep.disabled = c === 0;
  };
  const cards = groups.map((grp, g) => {
    const auto = t.auto.get(grp.key);
    const c = groupCard({
      list: auto ? auto.examples : grp.list, color: colors[g], editable: true, on: !t.off.has(grp.key),
      name: t.names.get(grp.key) ?? (auto ? auto.name : ''), share: grp.list.length / groups[0].list.length,
      words: auto && auto.words.length ? auto.words.join(', ') : '',
      onToggle: (v) => { if (v) t.off.delete(grp.key); else t.off.add(grp.key); repaint(); sync(); },
      onName: (v) => t.names.set(grp.key, v),
    });
    c.card.querySelector('.card-count').textContent = fmt(grp.list.length);
    c.card.addEventListener('pointerenter', () => { hover = { items: grp.list, color: colors[g] }; });
    c.nameEl.placeholder = 'Naming…';
    return c;
  });
  const actions = el('div', 'tray-actions');
  actions.append(keep, button('Cancel', 'is-quiet', closeTray));
  $('tray-body').replaceChildren(
    el('h2', 'tray-title', `Sorting ${t.where}`),
    el('p', 'tray-text', `${plural(t.m, 'work')}, grouped on their full vectors. Move the slider and watch the dots: fewer, larger groups to the left; more, finer groups to the right. Names are suggestions from the titles and Gemma’s notes.`),
    field('Granularity', slider, value), ...cards.map((c) => c.card), actions);
  sync();
  // names and central works for the groups not named yet (after the slider rests)
  clearTimeout(t.nameTimer);
  const missing = groups.filter((g) => !t.auto.has(g.key));
  if (!missing.length) return;
  t.nameTimer = setTimeout(async () => {
    let res;
    try { res = await api('names', { groups: groups.map((g) => g.list) }); } catch (err) { showError(err.message); return; }
    if (tray !== t || t.groups !== groups) return;
    res.groups.forEach((r, g) => {
      const a = { words: r.words, examples: r.examples, name: r.words.length ? cap(r.words[0]) : `Group ${g + 1}` };
      t.auto.set(groups[g].key, a);
      const c = cards[g];
      if (!t.names.has(groups[g].key) && document.activeElement !== c.nameEl) c.nameEl.value = a.name;
      c.nameEl.placeholder = a.name;
      c.wordsEl.textContent = a.words.join(', ');
      c.wordsEl.hidden = !a.words.length;
      c.pics.replaceChildren(...a.examples.map(thumb));
    });
  }, 220);
}

async function applyCluster() {
  const t = tray;
  if (t.groups.some((g) => !t.off.has(g.key) && !t.auto.has(g.key) && !(t.names.get(g.key) || '').trim())) {
    showError('The names are still loading. Try again in a moment.');
    return;
  }
  t.groups.forEach((grp, g) => {
    if (t.off.has(grp.key)) return;
    const name = (t.names.get(grp.key) || '').trim() || t.auto.get(grp.key).name;
    const node = kids(t.node).find((k) => k.name.toLowerCase() === name.toLowerCase())
      || addNode(name, t.node, 'cluster', t.colors[g]);
    addLabels(node.id, grp.list, { source: 'suggestion', method: 'cluster', review: 'accepted-in-bulk' });
  });
  const parent = t.node;
  closeTray();
  focus = parent;
  commit();
}

// ---------- step 3: split by words (EmbeddingGemma) ----------

function openSplit(nodeId) {
  openTray({ kind: 'split', node: nodeId, text: '', res: null, thr: 0.01, names: new Map(), off: new Set() }, 'Split by words', [...members(nodeId)]);
  renderSplit();
}

async function runSplit() {
  const t = tray;
  const words = t.text.split(',').map((w) => w.trim()).filter(Boolean);
  if (words.length < 2) { t.note = 'Give at least two words or phrases, separated by commas.'; renderSplit(); return; }
  t.note = 'Embedding the phrases…';
  renderSplit();
  try {
    t.res = await api('split', { members: [...members(t.node)], words });
    t.note = null;
    t.names = new Map();
    t.off = new Set();
  } catch (err) {
    t.res = null;
    t.note = err.message;
    if (err.status === 503) setTimeout(() => { if (tray === t && !t.res) runSplit(); }, 5000);
  }
  if (tray === t) renderSplit();
}

function renderSplit() {
  const t = tray, nd = ix.byId.get(t.node);
  const input = el('input');
  input.type = 'text'; input.id = 'split-words'; input.value = t.text; input.spellcheck = false; input.autocomplete = 'off';
  input.placeholder = 'gold coin, silver coin';
  input.addEventListener('input', () => { t.text = input.value; });
  input.addEventListener('keydown', (e) => { if (e.key === 'Enter') runSplit(); });
  const nodes = [el('h2', 'tray-title', `Splitting ${nd.name}`),
    el('p', 'tray-text', 'For when the shape of the cloud does not separate what you care about. EmbeddingGemma embeds each phrase, and each work goes to the phrase it is closest to.'),
    field('Words or phrases, separated by commas', input)];
  if (t.note) nodes.push(el('p', 'tray-text', t.note));
  const actions = el('div', 'tray-actions');
  if (!t.res) {
    actions.append(button('Split', 'is-primary', runSplit), button('Cancel', 'is-quiet', closeTray));
    nodes.push(actions);
    $('tray-body').replaceChildren(...nodes);
    preview = null; paint();
    setTimeout(() => input.focus(), 0);
    return;
  }
  const { members: mem, words, sims } = t.res;
  const groups = words.map((w, g) => ({ word: w, list: [], g }));
  const unsure = [];
  mem.forEach((i, r) => {
    let a = 0, b = -1;
    for (let g = 1; g < words.length; g++) if (sims[r][g] > sims[r][a]) a = g;
    for (let g = 0; g < words.length; g++) if (g !== a && (b < 0 || sims[r][g] > sims[r][b])) b = g;
    const margin = sims[r][a] - sims[r][b];
    (margin >= t.thr ? groups[a].list : unsure).push({ i, margin, g: a });
  });
  for (const grp of groups) grp.list.sort((x, y) => y.margin - x.margin);
  unsure.sort((x, y) => y.margin - x.margin);
  t.groups = groups; t.unsure = unsure;
  const colors = t.colors = freshColors(t.node, groups.length);
  const repaint = () => {
    const groupOf = new Int16Array(n).fill(-1);
    for (const grp of groups) for (const x of grp.list) groupOf[x.i] = t.off.has(grp.word) ? -2 : grp.g;
    for (const x of unsure) groupOf[x.i] = -2;
    preview = { groupOf, rgb: colors.map(rgbOf) };
    paint();
  };
  repaint();
  const slider = el('input');
  slider.type = 'range'; slider.min = 0; slider.max = 0.06; slider.step = 0.002; slider.value = t.thr; slider.id = 'split-margin';
  slider.addEventListener('input', () => { t.thr = +slider.value; renderSplit(); $('split-margin').focus(); });
  nodes.push(field('Leave a work unsure when its two best phrases are closer than', slider, el('span', 'field-value', t.thr.toFixed(3))));
  const keep = button('', 'is-primary', guard(applySplit));
  const sync = () => {
    const c = groups.filter((g) => g.list.length && !t.off.has(g.word)).length;
    keep.textContent = `Keep ${plural(c, 'group')} as labels`;
    keep.disabled = c === 0;
  };
  const most = Math.max(1, ...groups.map((g) => g.list.length), unsure.length);
  for (const grp of groups) {
    if (!grp.list.length) { nodes.push(el('p', 'card-words', `Nothing is closest to “${grp.word}”.`)); continue; }
    const list = grp.list.map((x) => x.i);
    nodes.push(groupCard({
      list, color: colors[grp.g], editable: true, on: !t.off.has(grp.word), name: t.names.get(grp.word) ?? cap(grp.word),
      share: list.length / most, words: `closest to “${grp.word}” · most certain first`,
      onToggle: (v) => { if (v) t.off.delete(grp.word); else t.off.add(grp.word); repaint(); sync(); },
      onName: (v) => t.names.set(grp.word, v),
    }).card);
  }
  if (unsure.length) {
    const c = groupCard({ list: unsure.map((x) => x.i), name: 'Unsure', share: unsure.length / most,
      words: 'Too close to call. These wait as suggestions on their nearest phrase.', onToggle: () => {} });
    c.box.style.visibility = 'hidden';   // nothing to tick: these are never labelled here
    nodes.push(c.card);
  }
  actions.append(keep, button('Split again', '', runSplit), button('Cancel', 'is-quiet', closeTray));
  nodes.push(actions);
  $('tray-body').replaceChildren(...nodes);
  sync();
}

async function applySplit() {
  const t = tray;
  const made = new Map();
  for (const grp of t.groups) {
    if (!grp.list.length || t.off.has(grp.word)) continue;
    const name = (t.names.get(grp.word) ?? cap(grp.word)).trim() || cap(grp.word);
    const node = kids(t.node).find((k) => k.name.toLowerCase() === name.toLowerCase()) || addNode(name, t.node, 'text-split', t.colors[grp.g]);
    made.set(grp.g, node.id);
    const at = stamp(), own = ix.own.get(node.id);
    for (const x of grp.list) if (!own.has(x.i)) S.labels.push({ i: x.i, node: node.id, source: 'suggestion', method: 'text-split', review: 'accepted-in-bulk', margin: +x.margin.toFixed(4), phrase: grp.word, at });
    reindex();
  }
  const at = stamp();
  for (const x of t.unsure) {
    const node = made.get(x.g);
    if (!node || ix.own.get(node).has(x.i) || S.pending.some((r) => r.i === x.i && r.node === node)) continue;
    S.pending.push({ i: x.i, node, method: 'text-split', margin: +x.margin.toFixed(4), at });
  }
  const parent = t.node;
  closeTray();
  focus = parent;
  commit();
}

// ---------- the tree beside the dots ----------

function renderTree() {
  const root = $('tree');
  const rows = [];
  const top = kids(null);
  const biggest = Math.max(1, ...top.map((k) => members(k.id).size));
  const row = (nd, depth) => {
    const b = el('button', 'tree-row');
    b.type = 'button';
    b.setAttribute('role', 'treeitem');
    b.setAttribute('aria-selected', String(focus === (nd ? nd.id : null)));
    b.style.paddingLeft = `${8 + depth * 18}px`;
    const count = nd ? members(nd.id).size : n;
    const cnt = el('span', 'tree-count', fmt(count));
    if (nd) {
      b.dataset.node = nd.id;
      b.style.setProperty('--c', nd.color);
      const waiting = ix.pending.get(nd.id).length;
      if (waiting) cnt.appendChild(el('span', 'tree-pending', `+${waiting}`));
    } else {
      b.classList.add('is-root');
    }
    b.append(el('span', 'tree-swatch'), el('span', 'tree-name', nd ? nd.name : 'All works'), cnt);
    if (nd) {
      const bar = el('span', 'tree-bar'), fill = el('i');
      const of = nd.parent != null ? members(nd.parent).size : biggest;
      fill.style.width = `${Math.max(1.5, 100 * count / Math.max(1, of))}%`;
      bar.appendChild(fill);
      b.appendChild(bar);
      b.addEventListener('pointerenter', () => { hover = { items: [...members(nd.id)].slice(0, 800), color: nd.color }; });
      b.addEventListener('pointerleave', () => { hover = null; });
    }
    b.addEventListener('click', () => setFocus(nd ? nd.id : null));
    rows.push(b);
    if (nd) for (const k of kids(nd.id)) row(k, depth + 1);
  };
  row(null, 0);
  if (!top.length) {
    rows.push(el('p', 'tree-empty', 'No labels yet. Zoom in on a cluster, draw around it, and give it a name. Or let the page propose a first set of groups with Sub-cluster.'));
  } else {
    const labelled = new Set(S.labels.map((r) => r.i)).size;
    rows.push(el('p', 'tree-note', `${fmt(n - labelled)} not labelled yet`));
    for (const k of top) row(k, 1);
  }
  root.replaceChildren(...rows);
  const labelled = new Set(S.labels.map((r) => r.i)).size;
  $('holdings').textContent = `${fmt(labelled)} of ${fmt(n)} works labelled · ${plural(S.nodes.length, 'label')}`;
}

function renderActions() {
  const box = $('node-actions');
  const nd = focus != null ? ix.byId.get(focus) : null;
  const title = el('h2', 'node-title');
  if (nd) { const sw = el('span', 'tree-swatch'); sw.style.setProperty('--c', nd.color); title.appendChild(sw); }
  title.appendChild(el('span', null, nd ? nd.name : 'All works'));
  const row = el('div', 'btn-row');
  let sub;
  if (!nd) {
    sub = S.pending.length ? `${plural(S.pending.length, 'suggestion')} waiting across the labels` : 'Dots are coloured by top-level label';
    row.appendChild(button(kids(null).length ? 'Sub-cluster the rest' : 'Sub-cluster', '', guard(() => openCluster(null))));
  } else {
    const waiting = ix.pending.get(nd.id).length;
    const loose = unsorted(nd.id).length;
    sub = `${plural(members(nd.id).size, 'work')}` + (kids(nd.id).length ? ` · ${fmt(loose)} not in a sub-label` : '') + (nd.parent != null ? ` · in ${pathOf(nd.parent)}` : '');
    row.append(
      button('Sub-cluster', '', guard(() => openCluster(nd.id))),
      button('Split by words', '', () => openSplit(nd.id)),
      button('Find more', '', guard(async () => { await findMore(nd.id); openReview(nd.id); })));
    if (waiting) row.appendChild(button(`Review ${waiting}`, 'is-primary', () => openReview(nd.id)));
    row.append(button('Rename', 'is-quiet', () => renameNode(nd)), button('Delete', 'is-quiet is-danger', () => {
      const under = descendants(nd.id).length - 1;
      if (!window.confirm(`Delete the label “${nd.name}”${under ? ` and its ${plural(under, 'sub-label')}` : ''}? The works stay; only the labels go.`)) return;
      if (tray) closeTray();
      deleteNode(nd.id);
      focus = nd.parent ?? null;
      commit();
    }));
  }
  box.replaceChildren(title, el('p', 'node-sub', sub), row);
}

function renameNode(nd) {
  const input = el('input', 'rename');
  input.type = 'text'; input.value = nd.name; input.spellcheck = false; input.setAttribute('aria-label', 'New name');
  const done = (ok) => {
    const v = input.value.trim();
    if (ok && v && v !== nd.name) { nd.name = v; commit(); } else renderActions();
  };
  input.addEventListener('keydown', (e) => { if (e.key === 'Enter') done(true); if (e.key === 'Escape') { e.stopPropagation(); done(false); } });
  input.addEventListener('blur', () => done(true));
  $('node-actions').querySelector('.node-title').replaceWith(input);
  input.focus();
  input.select();
}

// ---------- the placard: a work's labels, and where each came from ----------

const REVIEW_TEXT = { 'by-hand': 'selected by hand', accepted: 'suggested, accepted', 'accepted-in-bulk': 'suggested, accepted as a group' };

function openDetail(i) {
  tip.show(null);
  detail.open(i);
  renderDetailLabels(i);
}

function renderDetailLabels(i) {
  const box = $('detail-labels');
  const rows = [];
  for (const r of S.labels.filter((x) => x.i === i)) {
    const nd = ix.byId.get(r.node);
    const line = el('div', 'chip-row');
    const sw = el('span', 'tree-swatch'); sw.style.setProperty('--c', nd.color);
    line.append(sw, el('span', 'chip-path', pathOf(nd.id)), el('span', 'chip-src', `${REVIEW_TEXT[r.review]} · ${r.method}`),
      button('Remove', '', () => { removeLabel(i, nd.id); commit(); }));
    rows.push(line);
  }
  for (const r of S.pending.filter((x) => x.i === i)) {
    const nd = ix.byId.get(r.node);
    const line = el('div', 'chip-row');
    const sw = el('span', 'tree-swatch'); sw.style.setProperty('--c', nd.color);
    const yn = el('span', 'yesno');
    yn.style.marginLeft = 'auto';
    const yes = el('button', 'yes', '✓'), no = el('button', 'no', '✕');
    yes.type = no.type = 'button';
    yes.addEventListener('click', () => { decide(r, true); commit(); if (tray && tray.kind === 'review') renderReview(); });
    no.addEventListener('click', () => { decide(r, false); commit(); if (tray && tray.kind === 'review') renderReview(); });
    yn.append(yes, no);
    line.append(sw, el('span', 'chip-path', `${pathOf(nd.id)}?`), el('span', 'chip-src', 'suggested, waiting'), yn);
    rows.push(line);
  }
  if (rows.length) rows.unshift(el('p', 'detail-note-kicker', 'Labels'));
  box.replaceChildren(...rows);
}

// ---------- the select tool: draw around works, freehand ----------

function setArmed(v) {
  armed = v;
  $('tool-select').setAttribute('aria-pressed', String(v));
  document.body.classList.toggle('is-selecting', v);
  $('tools-hint').textContent = v
    ? 'Draw around the works you want · let go to close the shape · Esc to cancel'
    : 'or hold Shift and draw · drag to turn · scroll to zoom in on any spot';
}

function installSelect() {
  const stage = $('stage');
  // capture phase on the container: the press never reaches the cloud's own drag and orbit handlers
  stage.addEventListener('pointerdown', (e) => {
    if (e.button !== 0 || !(armed || e.shiftKey) || cloud.flying || cloud.view === 'morphing') return;
    e.stopPropagation();
    e.preventDefault();
    tip.show(null);
    lasso = [[e.clientX, e.clientY]];
    const move = (ev) => {
      if (!lasso) return;
      for (const pe of (ev.getCoalescedEvents ? ev.getCoalescedEvents() : [ev])) {
        const [lx, ly] = lasso[lasso.length - 1];
        if (Math.hypot(pe.clientX - lx, pe.clientY - ly) >= LASSO_STEP_PX) lasso.push([pe.clientX, pe.clientY]);
      }
    };
    const up = guard(async () => {
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
      const c = lasso;
      lasso = null;
      if (!c) return;
      setArmed(false);
      if (c.length < 3 || polyArea(c) < MIN_LASSO_AREA) return;
      const caught = caughtBy(c);
      if (!caught.length) { showError('No works inside that shape.'); return; }
      await openSelection(caught);
    });
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
  }, { capture: true });
  $('tool-select').addEventListener('click', () => setArmed(!armed));
  document.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape') return;
    if (lasso) { lasso = null; setArmed(false); return; }
    if (armed) { setArmed(false); return; }
    if (detail.isOpen) { detail.close(); return; }
    if (tray) closeTray();
  });
}

// ---------- start ----------

async function main() {
  const [itemsRes, pointsRes] = await Promise.all([fetch('data/items.json'), fetch('data/points.bin')]);
  for (const r of [itemsRes, pointsRes]) if (!r.ok) throw new Error(`${r.url}: HTTP ${r.status}`);
  items = await itemsRes.json();
  n = items.length;
  const buf = await pointsRes.arrayBuffer();
  if (buf.byteLength !== n * 15) throw new Error(`points.bin has ${buf.byteLength} bytes, expected ${n * 15}`);
  const rgb = new Uint8Array(buf, n * 12, n * 3);
  baseRgb = rgb.slice();
  cloud = new Cloud($('stage'), { xyz: new Float32Array(buf, 0, n * 3), rgb, items });
  cloud.controls.autoRotateSpeed = 0;      // a labelling surface holds still
  installPaneZoom();
  colorAttr = cloud.points.geometry.attributes.color;
  cloud.shift = shiftNow();
  cloud.applyShift();
  const home = cloud.framing(homePoints(), padsNow() / 2, HOME_FIT);   // open on the whole cloud, clear of the panel
  cloud.camera.position.copy(home.position);
  cloud.controls.target.copy(home.target);
  cloud.uniforms.uRef.value = home.dist;
  detail = new Detail(items, $('detail'), () => $('detail-labels').replaceChildren());
  tip = new Tip(items, $('dot-tip'));
  cloud.clickCb = (i) => openDetail(i);
  cloud.tipCb = (i, x, y) => tip.show(lasso ? null : i, x, y);
  reindex();
  try {
    Object.assign(S, await api('state', undefined, 'GET'));
  } catch (err) {   // a file saved for another bundle: start empty, say so, and leave the file alone until a change is made
    showError(`The saved labels were not loaded: ${err.message}`);
  }
  reindex();
  paint();
  renderTree();
  renderActions();
  installSelect();
  $('tray-close').addEventListener('click', closeTray);
  window.addEventListener('resize', settleShift);
  requestAnimationFrame(drawMarks);
  window.__cloud = cloud;
  window.__label = {
    get ready() { return ready; },
    get state() { return S; },
    get focus() { return focus; },
    get tray() { return tray ? tray.kind : null; },
    get view() { return cloud.view; },
    get flying() { return cloud.flying; },
    screenOf: (i) => screenOf(i),
    members: (id) => [...members(id)],
    caughtBy,
    cut,
  };
  await cloud.ready;
  $('loading').classList.add('is-done');
  setTimeout(() => { ready = true; }, 750);
}

main().catch((err) => {
  console.error(err);
  $('loading').textContent = `Could not load the collection: ${err.message}`;
  $('loading').classList.add('is-error');
});
