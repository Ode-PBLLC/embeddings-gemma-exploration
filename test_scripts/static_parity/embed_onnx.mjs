// Embed the parity queries with the ONNX build of EmbeddingGemma 2 (Transformers.js, Node, CPU).
//   node embed_onnx.mjs <dtype> [queries.json] [out_dir]
// Writes <out_dir>/onnx_<dtype>.f32 (rows = gallery queries then moon queries, 768 float32 each)
// and <out_dir>/onnx_<dtype>.json (query order, timings). Text encoder only, like the browser will load it.
import { AutoConfig, AutoModel, AutoTokenizer } from "@huggingface/transformers";
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";

const MODEL_ID = "onnx-community/embeddinggemma-2-ONNX";
const QUERY_PROMPT = "task: search result | query: "; // must equal gallery.embed.QUERY_PROMPT

const [dtype, qpath = "queries.json", outDir = "out"] = process.argv.slice(2);
if (!dtype) throw new Error("usage: node embed_onnx.mjs <dtype> [queries.json] [out_dir]");
const sets = JSON.parse(readFileSync(qpath, "utf8"));
const queries = [...sets.gallery, ...sets.moon].map((x) => x.q.trim());

const config = await AutoConfig.from_pretrained(MODEL_ID);
config.vision_config = config.audio_config = null; // text only
const t0 = performance.now();
const model = await AutoModel.from_pretrained(MODEL_ID, { config, device: "cpu", dtype });
const tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID);
const loadMs = performance.now() - t0;

const out = new Float32Array(queries.length * 768);
const ms = [];
for (let i = 0; i < queries.length; i++) {
  const t = performance.now();
  const { sentence_embedding } = await model(await tokenizer([QUERY_PROMPT + queries[i]]));
  ms.push(performance.now() - t);
  if (sentence_embedding.dims[1] !== 768) throw new Error(`dims ${sentence_embedding.dims}`);
  const v = sentence_embedding.data;
  let n = 0;
  for (let j = 0; j < 768; j++) n += v[j] * v[j];
  n = Math.sqrt(n);
  if (!Number.isFinite(n) || n === 0) throw new Error(`bad vector for ${queries[i]}`);
  for (let j = 0; j < 768; j++) out[i * 768 + j] = v[j] / n;
}
mkdirSync(outDir, { recursive: true });
writeFileSync(`${outDir}/onnx_${dtype}.f32`, Buffer.from(out.buffer));
const sorted = [...ms].sort((a, b) => a - b);
writeFileSync(`${outDir}/onnx_${dtype}.json`, JSON.stringify({
  dtype, n: queries.length, load_ms: Math.round(loadMs),
  median_ms: sorted[sorted.length >> 1], p95_ms: sorted[Math.floor(sorted.length * 0.95)], queries,
}, null, 1));
console.log(`${dtype}: ${queries.length} queries, load ${Math.round(loadMs)} ms, median ${sorted[sorted.length >> 1].toFixed(1)} ms`);
