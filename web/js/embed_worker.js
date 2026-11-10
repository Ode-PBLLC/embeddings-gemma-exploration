// EmbeddingGemma 2 (ONNX, Transformers.js) in a worker, so loading and inference never stall the cloud.
// Only the static site loads this (meta.static); the local server never fetches anything remote.
// Text encoder only. WebGPU runs the q8 build; without WebGPU, WASM runs fp16 (onnxruntime-web has no
// WASM kernel for the quantized builds). Both match the server's results: reports/static_parity.md.
//   in:  {type: 'load', model}  then  {type: 'embed', id, text}   (text already carries the query prompt)
//   out: {type: 'status', status}  and  {type: 'vector', id, vector} | {type: 'error', id, message}

const CACHE = 'transformers-cache';   // Transformers.js' browser cache
let ready = null;

const status = (s) => self.postMessage({ type: 'status', status: s });

async function hasWebGPU() {
  try {
    return !!(self.navigator.gpu && await self.navigator.gpu.requestAdapter());
  } catch {
    return false;
  }
}

async function isCached(model, files) {
  if (!self.caches) return false;
  try {
    const cache = await caches.open(CACHE);
    const big = Object.keys(files).find((f) => f.endsWith('.onnx_data'));
    return !!(await cache.match(`${model.hub}/${model.id}/resolve/${model.revision}/${big}`));
  } catch {
    return false;   // a blocked cache only means we can't say "from this browser"
  }
}

async function loadOn(tf, model, device) {
  const { dtype, files } = model.paths[device];
  const total = Object.values(files).reduce((a, b) => a + b, 0);
  const cached = await isCached(model, files);
  const loaded = {};
  status({ state: 'downloading', device, loaded: 0, total, cached });
  const progress_callback = (p) => {
    if (p.status === 'progress' && p.file in files) {
      loaded[p.file] = p.loaded;
      status({ state: 'downloading', device, loaded: Object.values(loaded).reduce((a, b) => a + b, 0), total, cached });
    }
  };
  const opts = { revision: model.revision, progress_callback };
  const config = await tf.AutoConfig.from_pretrained(model.id, opts);
  config.vision_config = config.audio_config = null;   // text only
  const tokenizer = await tf.AutoTokenizer.from_pretrained(model.id, opts);
  const net = await tf.AutoModel.from_pretrained(model.id, { ...opts, config, device, dtype });
  status({ state: 'preparing', device, loaded: total, total, cached });
  const embed = async (text) => {
    const { sentence_embedding } = await net(await tokenizer([text]));
    return sentence_embedding.data;
  };
  await embed('warm up');   // the first run compiles the WebGPU shaders
  return embed;
}

async function load(model) {
  const tf = await import(model.transformers_js);
  tf.env.allowLocalModels = false;   // weights come from the Hugging Face CDN at a pinned revision
  if (await hasWebGPU()) {
    try {
      return await loadOn(tf, model, 'webgpu');
    } catch (e) {
      console.warn('WebGPU load failed, falling back to WASM:', e);
    }
  }
  return loadOn(tf, model, 'wasm');
}

self.onmessage = async ({ data }) => {
  if (data.type === 'load') {
    ready = load(data.model);
    try {
      await ready;
      status({ state: 'ready' });
    } catch (e) {
      status({ state: 'failed', message: String(e && e.message || e) });
    }
    return;
  }
  if (data.type === 'embed') {
    try {
      const embed = await ready;
      const v = Float32Array.from(await embed(data.text));
      self.postMessage({ type: 'vector', id: data.id, vector: v.buffer }, [v.buffer]);
    } catch (e) {
      self.postMessage({ type: 'error', id: data.id, message: String(e && e.message || e) });
    }
  }
};
