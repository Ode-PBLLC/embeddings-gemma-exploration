// Search scope: deterministic metadata filters (type, collection) that sit beside the chat box.
// The semantic part of a query stays the embedding; the scope only decides which works are eligible.

const SOURCES = ['aic', 'nasa', 'wiki'];
const SHORT = { aic: 'AIC', nasa: 'NASA', wiki: 'Wikipedia' };

export class Scope {
  constructor(items, { onChange }) {
    this.items = items;
    this.onChange = onChange;
    this.modality = null;              // null = all types
    this.sources = new Set(SOURCES);
    this.btn = document.getElementById('scope-btn');
    this.panel = document.getElementById('scope');
    this.labelEl = document.getElementById('scope-label');
    this.countEl = document.getElementById('scope-count');
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

  // The request body's `filters`: only the fields that actually narrow the search.
  filters() {
    const f = {};
    if (this.modality) f.modality = [this.modality];
    if (this.sources.size < SOURCES.length) f.source = SOURCES.filter((s) => this.sources.has(s));
    return f;
  }

  matches(it) {
    return (!this.modality || it.modality === this.modality) && this.sources.has(it.source);
  }

  label() {
    const parts = [];
    if (this.modality) parts.push({ image: 'Images', text: 'Text', video: 'Video', audio: 'Sound' }[this.modality]);
    if (this.sources.size < SOURCES.length) parts.push(SOURCES.filter((s) => this.sources.has(s)).map((s) => SHORT[s]).join(' + '));
    return parts.length ? parts.join(' · ') + ' only' : 'All works';
  }

  update() {
    this.panel.querySelectorAll('[data-modality]').forEach((b) =>
      b.setAttribute('aria-checked', String((b.dataset.modality || null) === this.modality)));
    this.panel.querySelectorAll('[data-source]').forEach((b) =>
      b.setAttribute('aria-pressed', String(this.sources.has(b.dataset.source))));
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
