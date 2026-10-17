// The gallery wall (side panel) and the full page view.
import { SoundCard, pauseAllSounds } from './sound.js';

const SOURCE_NAME = { aic: 'Art Institute of Chicago', nasa: 'NASA Image Library', wiki: 'Wikipedia', moon: 'The Moon · LOLA via OpenPlanetaryMap' };

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}

function metaLine(it) {
  return [it.creator, it.date].filter(Boolean).join(', ');
}

// What the server actually filtered on, e.g. " · searched Text in Wikipedia" (empty when unfiltered).
function scopeNote(filters) {
  if (!filters || !Object.keys(filters).length) return '';
  const type = filters.modality ? ({ image: 'images', text: 'text', video: 'video', audio: 'sound' }[filters.modality[0]] || filters.modality[0]) : 'all types';
  const where = filters.source ? filters.source.map((s) => SOURCE_NAME[s] || s).join(' + ') : 'all collections';
  return ` · searched ${type} in ${where}`;
}

// One hung piece: a framed picture with a museum label, or a text card.
function piece(it, n, { large = false } = {}) {
  const pictured = it.modality === 'image' || it.modality === 'video';
  const kind = pictured ? 'image' : it.modality === 'audio' ? 'sound' : 'text';
  const fig = el('figure', `piece piece--${kind}${it.modality === 'video' ? ' piece--video' : ''}${large ? ' piece--large' : ''}`);
  fig.dataset.rank = String(n.rank);
  fig.dataset.index = String(n.index);
  fig.dataset.source = it.source;
  fig.style.setProperty('--work', it.color || '#9a958c');
  if (it.modality === 'audio') {
    // a listening station: waveform, karaoke transcript or static caption, title
    const card = new SoundCard(it, { size: large ? 'page' : 'wall' });
    fig.sound = card;
    const head = el('p', 'card-kicker');
    head.appendChild(el('span', 'label-no', `No. ${n.rank}`));
    head.appendChild(el('span', null, it.source === 'aic' ? 'Audio guide · Art Institute of Chicago' : 'Sound · NASA'));
    card.root.prepend(head);
    card.root.appendChild(el('h3', 'sound-title', it.title));
    const m = metaLine(it);
    if (m) card.root.appendChild(el('p', 'sound-meta', m));
    if ((it.license || '').startsWith('©')) card.root.appendChild(el('p', 'sound-notice', '© Art Institute of Chicago'));
    fig.appendChild(card.root);
  } else if (pictured) {
    const frame = el('div', 'frame');
    const mat = el('div', 'mat');
    if (it.modality === 'video' && it.video) {
      // a silent loop, playing on the wall like a screen in a gallery
      const v = el('video');
      Object.assign(v, { src: it.video, poster: it.thumb, muted: true, loop: true, autoplay: true, playsInline: true });
      v.setAttribute('muted', ''); v.setAttribute('playsinline', ''); v.preload = 'auto';
      v.setAttribute('aria-label', it.title);
      mat.appendChild(v);
    } else {
      const img = el('img');
      img.src = it.thumb;
      img.alt = it.title;
      img.loading = 'eager';
      img.decoding = 'async';
      mat.appendChild(img);
    }
    frame.appendChild(mat);
    fig.appendChild(frame);
    const cap = el('figcaption', 'label');
    cap.appendChild(el('span', 'label-no', `No. ${n.rank}`));
    cap.appendChild(el('span', 'label-title', it.title));
    const m = metaLine(it);
    if (m) cap.appendChild(el('span', 'label-meta', m));
    cap.appendChild(el('span', 'label-src', (SOURCE_NAME[it.source] || it.source) + (it.modality === 'video' ? ' · Video' : '')));
    fig.appendChild(cap);
  } else {
    const card = el('div', 'card');
    const head = el('p', 'card-kicker');
    head.appendChild(el('span', 'label-no', `No. ${n.rank}`));
    head.appendChild(el('span', null, `Wikipedia · ${it.topic || ''}`));
    card.appendChild(head);
    card.appendChild(el('h3', 'card-title', it.title));
    card.appendChild(el('p', 'card-text', it.summary));
    if (it.summary_pending) card.appendChild(el('p', 'pending', 'Summary pending: showing the source paragraph'));
    fig.appendChild(card);
  }
  return fig;
}

// Salon hang: rank 1 across the top, then each piece goes to the shortest column.
function hangInto(container, items, neighbors, cols) {
  container.replaceChildren();
  container.style.setProperty('--cols', cols);
  const first = neighbors[0];
  const lead = piece(items[first.index], first, { large: true });
  lead.classList.add('piece--lead');
  container.appendChild(lead);
  const row = el('div', 'hang-cols');
  const columns = Array.from({ length: cols }, () => { const c = el('div', 'hang-col'); row.appendChild(c); return c; });
  const heights = new Array(cols).fill(0);
  for (const n of neighbors.slice(1)) {
    const it = items[n.index];
    // rough height estimate so columns balance before images load
    const h = it.modality === 'audio' ? 0.95 : it.modality !== 'text' ? 1.25 : 0.6 + Math.min(it.summary.length, 600) / 450;
    const c = heights.indexOf(Math.min(...heights));
    heights[c] += h;
    columns[c].appendChild(piece(it, n));
  }
  container.appendChild(row);
  container.querySelectorAll('.piece').forEach((p, i) => p.style.setProperty('--i', i));
}

export class Wall {
  constructor({ items, onHover, onOpenPage, onBack, onClose }) {
    this.items = items;
    this.panel = document.getElementById('wall');
    this.page = document.getElementById('page');
    this.hang = document.getElementById('hang');
    this.pageHang = document.getElementById('page-hang');
    this.label = document.getElementById('wall-label');
    for (const root of [this.hang, this.pageHang]) {
      root.addEventListener('pointerover', (e) => {
        const p = e.target.closest('.piece');
        if (!p) return;
        onHover(Number(p.dataset.index));
        if (p.classList.contains('piece--image') || p.classList.contains('piece--sound')) this.showLabel(p);
        if (p.sound && !p.sound.playing) p.sound.play();   // sounds play while you rest on them
      });
      root.addEventListener('pointerout', (e) => {
        const p = e.target.closest('.piece');
        if (p && !p.contains(e.relatedTarget)) { onHover(null); this.hideLabel(); if (p.sound) p.sound.pause(); }
      });
      root.addEventListener('scroll', () => this.hideLabel(), { passive: true });
    }
    this.page.addEventListener('scroll', () => this.hideLabel(), { passive: true });
    document.getElementById('open-page').addEventListener('click', onOpenPage);
    document.getElementById('back-to-cloud').addEventListener('click', onBack);
    document.getElementById('close-wall').addEventListener('click', onClose);
  }

  render(result) {
    const { neighbors, query, weak, mode } = result;
    const srcs = new Set(neighbors.map((n) => this.items[n.index].source));
    const title = `“${query}”`;
    const kicker = weak ? 'Weak matches' : 'Nearest works';
    const sub = `${neighbors.length} works · ${srcs.size} ${srcs.size === 1 ? 'source' : 'sources'}`;
    for (const id of ['wall-title', 'page-title']) document.getElementById(id).textContent = title;
    for (const id of ['wall-kicker', 'page-kicker']) {
      const k = document.getElementById(id);
      k.textContent = kicker;
      k.classList.toggle('is-weak', !!weak);
    }
    for (const id of ['wall-sub', 'page-sub']) {
      document.getElementById(id).textContent = sub + (mode === 'quota' ? ' · balanced across sources' : '')
        + scopeNote(result.filters);
    }
    document.getElementById('wall-fine').hidden = mode !== 'quota';
    hangInto(this.hang, this.items, neighbors, 2);
    hangInto(this.pageHang, this.items, neighbors, 3);
    const list = document.getElementById('page-sources');
    list.replaceChildren();
    for (const n of neighbors) {
      const it = this.items[n.index];
      const li = el('li');
      li.appendChild(el('span', 'src-title', it.title));
      li.appendChild(document.createTextNode(` — ${SOURCE_NAME[it.source] || it.source}${it.creator && it.source !== 'wiki' ? `, ${it.creator}` : ''}. `));
      const a = el('a', null, it.source_url);
      a.href = it.source_url; a.target = '_blank'; a.rel = 'noopener';
      li.appendChild(a);
      list.appendChild(li);
    }
    this.hang.scrollTop = 0;
  }

  // The wall label: Gemma's note hung beside the piece, on the side with more room.
  showLabel(pieceEl) {
    const it = this.items[Number(pieceEl.dataset.index)];
    const lab = this.label;
    lab.querySelector('.wall-label-text').textContent = it.summary;
    lab.querySelector('.pending').hidden = !it.summary_pending;
    lab.style.setProperty('--work', it.color || '#9a958c');
    const r = (pieceEl.querySelector('.frame') || pieceEl.querySelector('.sound-card') || pieceEl).getBoundingClientRect();
    const w = lab.offsetWidth, h = lab.offsetHeight, gap = 18, vw = window.innerWidth, vh = window.innerHeight;
    const left = r.left - gap - w >= 12 ? r.left - gap - w : Math.min(r.right + gap, vw - w - 12);
    const top = Math.min(Math.max(r.top, 12), vh - h - 12);
    lab.style.transform = `translate(${Math.round(left)}px, ${Math.round(top)}px)`;
    lab.classList.add('is-shown');
  }

  hideLabel() {
    this.label.classList.remove('is-shown');
  }

  show() {
    this.panel.classList.add('is-open');
    this.panel.setAttribute('aria-hidden', 'false');
  }

  hide() {
    pauseAllSounds();
    this.hideLabel();
    this.panel.classList.remove('is-open');
    this.panel.setAttribute('aria-hidden', 'true');
    this.closePage();
  }

  openPage() {
    this.hideLabel();
    this.page.classList.add('is-open');
    this.page.setAttribute('aria-hidden', 'false');
    this.page.scrollTop = 0;
  }

  closePage() {
    this.hideLabel();
    this.page.classList.remove('is-open');
    this.page.setAttribute('aria-hidden', 'true');
  }

  get pageOpen() { return this.page.classList.contains('is-open'); }

  mark(index) {
    for (const p of document.querySelectorAll('.piece.is-linked')) p.classList.remove('is-linked');
    if (index == null) return;
    for (const p of document.querySelectorAll(`.piece[data-index="${index}"]`)) p.classList.add('is-linked');
  }

  panelWidth() {
    return this.panel.getBoundingClientRect().width;
  }
}
