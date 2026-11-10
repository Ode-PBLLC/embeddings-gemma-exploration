// Search scope: deterministic metadata filters (type, collection, artist, years) that sit beside the
// chat box. The semantic part of a query stays the embedding; the scope only decides which works are
// eligible. trail() describes the active filters for the breadcrumb.

const SOURCES = ['aic', 'nasa', 'wiki'];
const SHORT = { aic: 'AIC', nasa: 'NASA', wiki: 'Wikipedia' };
const LONG = { aic: 'Art Institute of Chicago', nasa: 'NASA', wiki: 'Wikipedia' };
const TYPE = { image: 'Images', text: 'Text', video: 'Video', audio: 'Sound' };

export const fmtYear = (y) => (y < 0 ? `${Math.round(-y)} BCE` : String(Math.round(y)));

export class Scope {
  constructor(items, { onChange }) {
    this.items = items;
    this.onChange = onChange;
    this.modality = null;              // null = all types
    this.sources = new Set(SOURCES);
    this.artist = null;                // one exact artist name, or null
    this.years = null;                 // [from, to] inclusive, or null
    const $ = (id) => document.getElementById(id);
    this.btn = $('scope-btn');
    this.panel = $('scope');
    this.labelEl = $('scope-label');
    this.countEl = $('scope-count');
    this.artistEl = $('scope-artist');
    this.artistList = $('scope-artists');
    this.fromEl = $('scope-from');
    this.toEl = $('scope-to');
    // a bundle without dates or artists (e.g. the synthetic set) simply has no such rows
    const dated = items.filter((it) => it.years);
    this.fromEl.closest('.scope-group').hidden = dated.length === 0;
    this.artistEl.closest('.scope-group').hidden = !items.some((it) => it.artist);
    this.span = dated.length ? [Math.min(...dated.map((it) => it.years[0])), Math.max(...dated.map((it) => it.years[1]))] : [0, 0];
    this.fromEl.placeholder = fmtYear(this.span[0]);
    this.toEl.placeholder = fmtYear(this.span[1]);

    this.btn.addEventListener('click', () => this.toggle());
    this.panel.querySelectorAll('[data-modality]').forEach((b) => b.addEventListener('click', () => {
      this.modality = b.dataset.modality || null;
      this.update();
    }));
    this.panel.querySelectorAll('[data-source]').forEach((b) => b.addEventListener('click', () => {
      const s = b.dataset.source;
      if (this.sources.has(s)) {
        if (this.sources.size > 1) this.sources.delete(s);  // keep at least one collection
      } else this.sources.add(s);
      this.update();
    }));
    // artist: only an exact name from the list counts; anything else is shown as not matching
    this.artistEl.addEventListener('change', () => this.setArtist(this.artistEl.value.trim() || null));
    this.artistEl.addEventListener('input', () => {
      const v = this.artistEl.value.trim();
      this.artistEl.classList.toggle('is-invalid', !!v && !this.artistCounts().has(v));
      if (this.artistCounts().has(v)) this.setArtist(v);   // picked from the list
    });
    const readYears = () => {
      const a = this.fromEl.value === '' ? null : Number(this.fromEl.value);
      const b = this.toEl.value === '' ? null : Number(this.toEl.value);
      if (a == null && b == null) return this.setYears(null);
      const lo = a ?? this.span[0], hi = b ?? this.span[1];
      const bad = !(Number.isFinite(lo) && Number.isFinite(hi)) || lo > hi;
      this.fromEl.classList.toggle('is-invalid', bad);
      this.toEl.classList.toggle('is-invalid', bad);
      if (!bad) this.setYears([lo, hi]);
    };
    this.fromEl.addEventListener('change', readYears);
    this.toEl.addEventListener('change', readYears);
    document.getElementById('scope-clear').addEventListener('click', () => this.clear());
    // Enter in a scope field applies it (via its change event); it never submits the search
    for (const el of [this.artistEl, this.fromEl, this.toEl]) {
      el.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); el.dispatchEvent(new Event('change')); } });
    }
    document.addEventListener('pointerdown', (e) => {
      if (!this.panel.hidden && !this.panel.contains(e.target) && !this.btn.contains(e.target)) this.toggle(false);
    });
    this.update();
  }

  get isOpen() { return !this.panel.hidden; }

  toggle(open = this.panel.hidden) {
    this.panel.hidden = !open;
    this.btn.setAttribute('aria-expanded', String(open));
  }

  setArtist(name) {
    this.artist = name;
    this.update();
  }

  setYears(range) {
    this.years = range;
    this.update();
  }

  clear() {
    this.modality = null;
    this.sources = new Set(SOURCES);
    this.artist = null;
    this.years = null;
    this.update();
  }

  // The request body's `filters`: only the fields that actually narrow the search.
  filters() {
    const f = {};
    if (this.modality) f.modality = [this.modality];
    if (this.sources.size < SOURCES.length) f.source = SOURCES.filter((s) => this.sources.has(s));
    if (this.artist) f.artist = [this.artist];
    if (this.years) f.year = [...this.years];
    return f;
  }

  matchesBase(it) {   // type and collection: what the artist list is drawn from
    return (!this.modality || it.modality === this.modality) && this.sources.has(it.source);
  }

  matches(it) {
    if (!this.matchesBase(it)) return false;
    if (this.artist && it.artist !== this.artist) return false;
    if (this.years && !(it.years && it.years[1] >= this.years[0] && it.years[0] <= this.years[1])) return false;
    return true;
  }

  // artist -> works, among the works the type and collection allow
  artistCounts() {
    const m = new Map();
    for (const it of this.items) if (it.artist && this.matchesBase(it)) m.set(it.artist, (m.get(it.artist) || 0) + 1);
    return m;
  }

  label() {
    const parts = [];
    if (this.modality) parts.push(TYPE[this.modality]);
    if (this.sources.size < SOURCES.length) parts.push(SOURCES.filter((s) => this.sources.has(s)).map((s) => SHORT[s]).join(' + '));
    if (this.artist) parts.push(this.artist);
    if (this.years) parts.push(`${fmtYear(this.years[0])}–${fmtYear(this.years[1])}`);
    return parts.length ? parts.join(' · ') + ' only' : 'All works';
  }

  // The breadcrumb: one step per active filter, in drill-down order, each with a way to remove it.
  trail() {
    const steps = [];
    if (this.sources.size < SOURCES.length) {
      steps.push({ kind: 'metadata', field: 'Collection', label: SOURCES.filter((s) => this.sources.has(s)).map((s) => LONG[s]).join(' + '),
        clear: () => { this.sources = new Set(SOURCES); this.update(); } });
    }
    if (this.modality) steps.push({ kind: 'metadata', field: 'Type', label: TYPE[this.modality], clear: () => { this.modality = null; this.update(); } });
    if (this.artist) steps.push({ kind: 'metadata', field: 'Artist', label: this.artist, clear: () => this.setArtist(null) });
    if (this.years) {
      steps.push({ kind: 'metadata', field: 'Years', label: `${fmtYear(this.years[0])}–${fmtYear(this.years[1])}`,
        clear: () => this.setYears(null) });
    }
    return steps;
  }

  update() {
    this.panel.querySelectorAll('[data-modality]').forEach((b) =>
      b.setAttribute('aria-checked', String((b.dataset.modality || null) === this.modality)));
    this.panel.querySelectorAll('[data-source]').forEach((b) =>
      b.setAttribute('aria-pressed', String(this.sources.has(b.dataset.source))));
    const counts = this.artistCounts();
    if (this.artist && !counts.has(this.artist)) this.artist = null;   // narrowed away by type/collection
    this.artistList.replaceChildren(...[...counts].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0])).map(([name, n]) => {
      const o = document.createElement('option');
      o.value = name;
      o.label = `${n} work${n === 1 ? '' : 's'}`;
      return o;
    }));
    if (document.activeElement !== this.artistEl) this.artistEl.value = this.artist || '';
    this.artistEl.classList.remove('is-invalid');
    if (!this.years) { this.fromEl.value = this.toEl.value = ''; }
    this.fromEl.classList.remove('is-invalid');
    this.toEl.classList.remove('is-invalid');
    const flags = new Float32Array(this.items.length);
    let n = 0;
    this.items.forEach((it, i) => { if (this.matches(it)) n++; else flags[i] = 1; });
    this.count = n;
    this.countEl.textContent = n ? `${n.toLocaleString()} works in scope` : 'No works match this scope';
    this.countEl.classList.toggle('is-empty', n === 0);
    this.labelEl.textContent = this.label();
    this.btn.classList.toggle('is-filtered', Object.keys(this.filters()).length > 0);
    if (this.onChange) this.onChange(flags);
  }
}
