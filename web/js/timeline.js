// Timeline view: the works leave the cloud for a plane square to the camera (through the current
// target), with time running left to right, and pile up by year as a coloured beeswarm. Only dated
// works in the search scope take part (the Wikipedia passages have no date); the rest drop out, and
// changing the scope re-stacks the swarm. The axis is compressed toward the past so that NASA's
// decades and four thousand years of art both stay readable.
import * as THREE from 'three';
import { tween, ease } from './anim.js';
import { fmtYear } from './scope.js';

const NOW = 2027;            // the axis's right edge (years before this are compressed)
const SOFT = 30;             // years: below this the axis is close to linear, beyond it logarithmic
const WIDTH = 0.84;          // share of the visible width the axis spans
const BASE = 0.30;           // baseline sits this share of the visible height below the target
const TALL = 0.60;           // piles rise at most this share of the visible height...
const FILL = 0.30;           // ...and the dots together cover about this share of the chart's area
const MIN_COLS = 160;        // never coarser than this many columns, so a small set keeps its dates
const MORPH_MS = 1100;
const TICKS = [-4000, -2000, -1000, -500, 1, 500, 1000, 1200, 1400, 1500, 1600, 1700, 1800, 1850, 1900,
  1925, 1950, 1975, 2000, 2010, 2020];
const TICK_GAP_PX = 58;

export class Timeline {
  constructor(cloud, items, { inScope }) {
    this.cloud = cloud;
    this.items = items;
    this.inScope = inScope;   // (index) => bool
    this.t = 0;               // 0 = cloud, 1 = timeline
    this.busy = false;
    this.mid = items.map((it) => (it.years ? (it.years[0] + it.years[1]) / 2 : null));
    const dated = this.mid.filter((y) => y != null);
    this.minYear = Math.min(...dated);
    this.goal = new Float32Array(cloud.n * 3);
    this.from = new Float32Array(cloud.n * 3);
    this.axisEl = document.getElementById('timeline-axis');
    const mat = new THREE.LineBasicMaterial({ color: 0x8a847a, transparent: true, opacity: 0, depthTest: false });
    this.axis = new THREE.LineSegments(new THREE.BufferGeometry(), mat);
    this.axis.renderOrder = 5;
    this.axis.visible = false;
    cloud.scene.add(this.axis);
    cloud.timeline = this;
  }

  get active() { return this.t > 0 || this.busy; }

  // 0..1 along the axis: log of (years before NOW + SOFT), so recent decades get room
  u(year) {
    const span = Math.log((NOW - this.minYear + SOFT) / SOFT);
    return 1 - Math.log((NOW - year + SOFT) / SOFT) / span;
  }

  // The plane: through the orbit target, square to the view, sized to what the camera sees there.
  frame() {
    const c = this.cloud, cam = c.camera;
    cam.updateMatrixWorld();
    const m = cam.matrixWorld.elements;
    const right = new THREE.Vector3(m[0], m[1], m[2]).normalize();
    const up = new THREE.Vector3(m[4], m[5], m[6]).normalize();
    const centre = c.controls.target.clone();
    const dist = cam.position.distanceTo(centre);
    const h = 2 * dist * Math.tan(THREE.MathUtils.degToRad(cam.fov) / 2);
    const w = h * cam.aspect;
    return { right, up, centre, w, h };
  }

  // Goal positions for the scope's dated works: a beeswarm of columns one dot wide, stacked up from
  // the baseline. Dot size comes from the total area (so one crowded date cannot shrink every dot);
  // a column that reaches the height limit spills into the nearest column with room, like sand.
  layout() {
    const f = this.plane, n = this.cloud.n;
    const members = [];
    for (let i = 0; i < n; i++) if (this.mid[i] != null && this.inScope(i)) members.push(i);
    const width = f.w * WIDTH, tall = f.h * TALL;
    const d = Math.min(width / MIN_COLS, Math.sqrt((width * tall * FILL) / Math.max(1, members.length)));
    const cols = Math.max(1, Math.floor(width / d)), cap = Math.max(1, Math.floor(tall / d));
    this.spacing = d;
    const home = (i) => Math.min(cols - 1, Math.floor(this.u(this.mid[i]) * cols));
    // place in time order, each work in its own column or, if that is full, the nearest one with room
    const order = [...members].sort((a, b) => this.mid[a] - this.mid[b]);
    const height = new Int32Array(cols);
    const byCol = new Map();
    for (const i of order) {
      const k0 = home(i);
      let k = k0;
      for (let r = 0; height[k] >= cap && r < cols; r++) {
        const right = k0 + r + 1, left = k0 - r - 1;
        if (right < cols && height[right] < cap) { k = right; break; }
        if (left >= 0 && height[left] < cap) { k = left; break; }
      }
      height[k]++;
      if (!byCol.has(k)) byCol.set(k, []);
      byCol.get(k).push(i);
    }
    const hue = (i) => {   // pile each column by collection, then colour, so the swarm shows bands
      const c = this.cloud.points.geometry.attributes.color;
      const r = c.getX(i), g = c.getY(i), b = c.getZ(i);
      return Math.atan2(Math.sqrt(3) * (g - b), 2 * r - g - b);
    };
    const x0 = -width / 2, base = -f.h * BASE;
    const P = new THREE.Vector3();
    for (const [k, list] of byCol) {
      list.sort((a, b) => (this.items[a].source < this.items[b].source ? -1 : this.items[a].source > this.items[b].source ? 1 : hue(a) - hue(b)));
      list.forEach((i, j) => {
        P.copy(f.centre).addScaledVector(f.right, x0 + (k + 0.5) * d).addScaledVector(f.up, base + (j + 0.5) * d);
        this.goal[i * 3] = P.x; this.goal[i * 3 + 1] = P.y; this.goal[i * 3 + 2] = P.z;
      });
    }
    this.members = new Set(members);
    this.drawAxis();
    return members.length;
  }

  drawAxis() {
    const f = this.plane, width = f.w * WIDTH, base = -f.h * BASE;
    const at = (u, dy = 0) => f.centre.clone().addScaledVector(f.right, -width / 2 + u * width).addScaledVector(f.up, base + dy);
    const pts = [at(0), at(1)];
    const tick = f.h * 0.008;
    this.ticks = TICKS.filter((y) => y >= this.minYear).map((y) => ({ y, p: at(this.u(y)) }));
    for (const t of this.ticks) pts.push(t.p, at(this.u(t.y), -tick));
    this.axis.geometry.setFromPoints(pts);
  }

  // Where a query sits on the timeline: above the swarm, at the similarity-weighted mean position of
  // its neighbours along the axis (the cloud uses the same weighting in 3D). Returns a world point.
  queryPoint(neighbors) {
    const f = this.plane, c = this.cloud;
    let sw = 0, sx = 0;
    for (const nb of neighbors) {
      const w = Math.max(nb.similarity, 0);
      const p = c.worldPos(nb.index).sub(f.centre);
      sx += w * p.dot(f.right); sw += w;
    }
    const x = sw > 0 ? sx / sw : 0;
    return f.centre.clone().addScaledVector(f.right, x).addScaledVector(f.up, -f.h * BASE + f.h * TALL + f.h * 0.06);
  }

  // The chart's corners (axis ends, top of the tallest allowed pile), for framing it beside the wall.
  bounds() {
    const f = this.plane, width = f.w * WIDTH, base = -f.h * BASE;
    const at = (x, y) => f.centre.clone().addScaledVector(f.right, x).addScaledVector(f.up, y);
    return [at(-width / 2, base - f.h * 0.05), at(width / 2, base - f.h * 0.05),
      at(-width / 2, base + f.h * (TALL + 0.08)), at(width / 2, base + f.h * (TALL + 0.08))];
  }

  // Each frame: place the tick labels under their ticks (the axis is part of the scene, so it pans and zooms).
  step() {
    if (!this.axis.visible) return;
    const shown = [];
    let last = -Infinity;
    for (const t of this.ticks) {
      const s = this.cloud.project(t.p);
      if (s.z > 1 || s.x - last < TICK_GAP_PX || s.x < 8 || s.x > window.innerWidth - 8) continue;
      last = s.x;
      shown.push([fmtYear(t.y), s.x, s.y]);
    }
    const key = shown.map((s) => s.join()).join('|');
    if (key === this.tickKey) return;
    this.tickKey = key;
    this.axisEl.replaceChildren(...shown.map(([label, x, y]) => {
      const e = document.createElement('span');
      e.textContent = label;
      e.style.transform = `translate(${x}px, ${y + 10}px) translateX(-50%)`;
      return e;
    }));
  }

  setAxisOpacity(a) {
    this.axis.material.opacity = 0.7 * a;
    this.axis.visible = a > 0.01;
    this.axisEl.style.opacity = String(a);
    if (!this.axis.visible) { this.axisEl.replaceChildren(); this.tickKey = null; }
  }

  // Move every taking-part work from where it is drawn now to its goal; the rest drop out.
  async glide(toTimeline, run) {
    const c = this.cloud, s = c.shown, n = c.n;
    this.from.set(s);
    const target = toTimeline ? this.goal : c.positions;
    for (let i = 0; i < n; i++) {
      const on = toTimeline ? this.members.has(i) : true;
      c.away[i] = on ? 0 : 2;     // 2 = not drawn
    }
    c.points.geometry.attributes.aAway.needsUpdate = true;
    const a0 = this.axis.material.opacity / 0.7;
    const done = await tween(MORPH_MS, (e) => {
      for (let k = 0; k < n * 3; k++) s[k] = this.from[k] + (target[k] - this.from[k]) * e;
      c.points.geometry.attributes.position.needsUpdate = true;
      this.setAxisOpacity(toTimeline ? Math.max(a0, e) : a0 * (1 - e));
    }, { run, easing: ease.inOutCubic });
    return done;
  }

  async enter(run = 0) {
    if (this.busy || this.t === 1) return;
    const c = this.cloud;
    this.busy = true;
    c.mode = 'timeline';
    c.controls.autoRotate = false;
    c.controls.enableRotate = false;   // the timeline is a flat chart: pan and zoom only
    this.plane = this.frame();
    this.home = { position: c.camera.position.clone(), target: c.controls.target.clone() };
    this.layout();
    this.setAxisOpacity(0.001);
    await this.glide(true, run);
    this.t = 1;
    this.busy = false;
  }

  // The scope changed while on the timeline: re-stack the works now in scope. A change that arrives
  // mid-glide is kept and applied when the glide lands, so quick successive filters all take effect.
  async restack(run = 0) {
    if (this.t !== 1) return;
    if (this.busy) { this.pending = run; return; }
    this.busy = true;
    this.layout();
    await this.glide(true, run);
    this.busy = false;
    if (this.pending != null) {
      const next = this.pending;
      this.pending = null;
      await this.restack(next);
    }
  }

  async exit(run = 0) {
    if (this.t === 0 && !this.busy) return;
    const c = this.cloud;
    this.busy = true;
    await this.glide(false, run);
    c.shown.set(c.positions);
    c.points.geometry.attributes.position.needsUpdate = true;
    c.away.fill(0);
    c.points.geometry.attributes.aAway.needsUpdate = true;
    this.setAxisOpacity(0);
    c.controls.enableRotate = true;
    c.mode = 'idle';
    c.lastInteract = performance.now();
    this.t = 0;
    this.busy = false;
  }
}
