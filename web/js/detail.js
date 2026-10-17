// The placard that opens when a dot or piece is clicked, and the small name tag that follows the pointer.
import { SoundCard, pauseAllSounds } from './sound.js';

const SOURCE_NAME = { aic: 'Art Institute of Chicago', nasa: 'NASA Image Library', wiki: 'Wikipedia', moon: 'The Moon · LOLA via OpenPlanetaryMap' };

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}

export class Detail {
  constructor(items, root, onClose) {
    this.items = items;
    this.root = root;
    this.onClose = onClose;
    this.index = null;
    root.querySelector('.detail-close').addEventListener('click', () => this.close());
  }

  get isOpen() { return this.index != null; }

  open(index) {
    const it = this.items[index];
    this.index = index;
    const body = this.root.querySelector('.detail-body');
    body.replaceChildren();
    this.root.dataset.source = it.source;
    this.root.style.setProperty('--work', it.color || '#9a958c');
    if (it.modality === 'audio' && it.audio) {
      const card = new SoundCard(it, { size: 'placard' });
      body.appendChild(card.root);
      card.play();   // opened by a click, so the browser allows sound
    } else if (it.modality === 'video' && it.video) {
      const frame = el('div', 'frame');
      const mat = el('div', 'mat');
      const v = el('video');
      Object.assign(v, { src: it.video, poster: it.thumb, muted: true, loop: true, autoplay: true, playsInline: true, controls: true });
      v.setAttribute('muted', ''); v.setAttribute('playsinline', '');
      v.setAttribute('aria-label', it.title);
      mat.appendChild(v);
      frame.appendChild(mat);
      body.appendChild(frame);
    } else if (it.modality === 'image' && it.thumb) {
      const frame = el('div', 'frame');
      const mat = el('div', 'mat');
      const img = el('img');
      img.alt = it.title;
      img.src = it.thumb;  // show the thumbnail at once, then swap in the larger image
      const big = it.thumb.replace(/^thumbs\//, 'media/');
      const probe = new Image();
      probe.onload = () => { if (this.index === index) img.src = big; };
      probe.onerror = () => console.warn('larger image unavailable, keeping thumbnail', big);
      probe.src = big;
      mat.appendChild(img);
      frame.appendChild(mat);
      body.appendChild(frame);
    }
    body.appendChild(el('p', 'detail-kicker', (SOURCE_NAME[it.source] || it.source) + ({ video: ' · Video', audio: it.source === 'aic' ? ' · Audio guide' : ' · Sound' }[it.modality] || '')));
    body.appendChild(el('h3', 'detail-title', it.title));
    const meta = [it.creator, it.date].filter(Boolean).join(', ');
    if (meta) body.appendChild(el('p', 'detail-meta', meta));
    body.appendChild(el('p', 'detail-note-kicker', it.modality === 'text' ? 'Gemma’s summary' : 'Gemma’s note'));
    body.appendChild(el('p', 'detail-text', it.summary));
    if (it.summary_pending) body.appendChild(el('p', 'pending', 'Summary pending: showing the source text'));
    if (it.modality === 'audio' && it.license) body.appendChild(el('p', 'detail-license', it.license));
    if (it.source_url) {
      const a = el('a', 'text-btn detail-link', { video: 'Watch the full video at NASA', audio: 'Hear it in context at NASA' }[it.modality] || 'View at source');
      a.href = it.source_url; a.target = '_blank'; a.rel = 'noopener';
      body.appendChild(a);
    }
    this.root.classList.add('is-open');
    this.root.setAttribute('aria-hidden', 'false');
  }

  close() {
    if (this.index == null) return;
    pauseAllSounds();
    this.index = null;
    this.root.classList.remove('is-open');
    this.root.setAttribute('aria-hidden', 'true');
    if (this.onClose) this.onClose();
  }
}

export class Tip {
  constructor(items, root) {
    this.items = items;
    this.root = root;
    this.index = null;
  }

  show(index, x, y) {
    if (index == null) { this.root.classList.remove('is-shown'); this.index = null; return; }
    if (index !== this.index) {
      const it = this.items[index];
      this.root.replaceChildren(el('span', 'tip-title', it.title), el('span', 'tip-src', SOURCE_NAME[it.source] || it.source));
      this.root.style.setProperty('--work', it.color || '#9a958c');
      this.index = index;
    }
    const w = this.root.offsetWidth || 220;
    const left = x + 16 + w > window.innerWidth ? x - 16 - w : x + 16;
    this.root.style.transform = `translate(${Math.round(left)}px, ${Math.round(y + 14)}px)`;
    this.root.classList.add('is-shown');
  }
}
