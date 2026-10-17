// Listening cards: a sound work's waveform, its karaoke transcript (or a static caption when it
// has no speech), and playback. Used on the wall, on the page and in the placard.

let current = null;   // the one SoundCard playing; starting another pauses it

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}

// Lower bound: index of the last word starting at or before t (or -1).
function wordAt(words, t) {
  let lo = 0, hi = words.length - 1, ans = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (words[mid][1] <= t) { ans = mid; lo = mid + 1; } else hi = mid - 1;
  }
  return ans;
}

export class SoundCard {
  // it: an items.json entry with audio, peaks, words, caption, color. size: 'wall' | 'page' | 'placard'
  constructor(it, { size = 'wall' } = {}) {
    this.it = it;
    this.words = it.words || [];
    this.root = el('div', `sound-card sound-card--${size}`);
    this.root.style.setProperty('--work', it.color || '#c98a2b');
    this.wave = el('canvas', 'sound-wave');
    this.wave.setAttribute('aria-hidden', 'true');
    this.btn = el('button', 'sound-play');
    this.btn.type = 'button';
    this.btn.setAttribute('aria-label', `Play ${it.title}`);
    const top = el('div', 'sound-top');
    top.append(this.btn, this.wave);
    this.line = el('div', 'sound-line');
    if (this.words.length) {
      this.track = el('p', 'sound-words');
      this.spans = this.words.map(([w, s], i) => {
        const sp = el('span', 'w', w + ' ');
        sp.dataset.i = String(i);
        sp.addEventListener('click', (e) => { e.stopPropagation(); this.seek(s); this.play(); });
        return sp;
      });
      this.track.append(...this.spans);
      this.line.append(this.track);
    } else {
      this.line.classList.add('is-caption');
      this.line.append(el('p', 'sound-caption', it.caption || it.title));
    }
    this.root.append(top, this.line);
    this.audio = new Audio();
    this.audio.preload = 'none';
    this.audio.src = it.audio;
    this.audio.addEventListener('ended', () => this.stop());
    this.btn.addEventListener('click', (e) => { e.stopPropagation(); this.toggle(); });
    this.wave.addEventListener('click', (e) => {
      e.stopPropagation();
      const r = this.wave.getBoundingClientRect();
      const dur = this.audio.duration || it.duration || 0;
      if (dur) { this.seek(dur * (e.clientX - r.left) / r.width); this.play(); }
    });
    this.active = -1;
    requestAnimationFrame(() => this.draw(0));
  }

  get playing() { return !this.audio.paused; }

  play() {
    if (current && current !== this) current.pause();
    current = this;
    this.root.classList.add('is-playing');
    this.audio.play().catch((e) => {
      // browsers refuse sound before the first click or key press on the page; say so on the card
      this.root.classList.remove('is-playing');
      this.root.classList.add('needs-gesture');
      console.warn('sound did not start', this.it.audio, e.message);
    });
    if (!this.raf) this.tick();
  }

  pause() {
    this.audio.pause();
    this.root.classList.remove('is-playing');
  }

  stop() { this.pause(); }

  toggle() { this.playing ? this.pause() : this.play(); }

  seek(t) { this.audio.currentTime = Math.max(0, t); this.update(); }

  tick() {
    this.update();
    this.raf = this.playing ? requestAnimationFrame(() => this.tick()) : null;
  }

  update() {
    const dur = this.audio.duration || this.it.duration || 1;
    this.draw(this.audio.currentTime / dur);
    if (!this.spans) return;
    const i = wordAt(this.words, this.audio.currentTime);
    if (i === this.active) return;
    this.spans.forEach((sp, k) => { sp.classList.toggle('is-said', k < i); sp.classList.toggle('is-now', k === i); });
    this.active = i;
    const sp = this.spans[Math.max(0, i)];
    if (sp) {   // keep the current word in the middle of the line
      const box = this.line.getBoundingClientRect(), r = sp.getBoundingClientRect();
      this.line.scrollTop += (r.top + r.height / 2) - (box.top + box.height / 2);
    }
  }

  // Waveform: one bar per peak, played part in the work's colour, the rest pale.
  draw(progress) {
    const c = this.wave, dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = c.clientWidth || 260, h = c.clientHeight || 44;
    if (c.width !== Math.round(w * dpr)) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); }
    const g = c.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);
    const peaks = this.it.peaks || [];
    const n = peaks.length || 1, bw = w / n;
    const color = getComputedStyle(this.root).getPropertyValue('--work').trim() || '#c98a2b';
    for (let k = 0; k < n; k++) {
      const v = Math.max(0.06, (peaks[k] || 0) / 255);
      const bh = v * (h - 4);
      g.fillStyle = k / n <= progress ? color : 'rgba(246,244,239,0.28)';
      g.fillRect(k * bw + bw * 0.18, (h - bh) / 2, Math.max(1, bw * 0.64), bh);
    }
  }
}

export function pauseAllSounds() { if (current) current.pause(); }
