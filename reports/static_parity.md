# Static site: browser search vs the server

Measured 2026-11-10. The static site (GitHub Pages) embeds the query in the browser with the ONNX build of
EmbeddingGemma 2 (`onnx-community/embeddinggemma-2-ONNX` at revision `daa72c5`, Transformers.js 4.3.2,
onnxruntime-web 1.31.0-dev) and searches the stored document vectors in JavaScript. The stored vectors came from the
full model in bf16 on the GPU. This report measures how closely the browser's answers match the running server.

## Method

- **Queries:** 281 in total.
  - Gallery, 246: the 158 prompts of `reports/demo_prompt_sweep.md`, 8 more on-topic prompts from
    `eval/queries.yaml`, 50 specific queries (artists, missions, objects, questions) and 30 nothing-close queries
    (the 10 in `eval/queries.yaml` plus 20 more).
  - Moon, 35: the 15 on-topic and 10 off-topic prompts in `eval/moon_queries.yaml`, plus 10 more.
  - The list is in `test_scripts/static_parity/queries.json`.
- **Reference:** the live server's `POST /search` answer for every query (`:8000`, GPU, bf16, all 5,558 gallery
  items).
- **Candidates:** query vectors from:
  - Node (onnxruntime-node, CPU), for every dtype;
  - Windows Edge with the real browser runtimes (WebGPU on an RTX 3070 Ti, and WASM);
  - the server's own code on CPU (bf16 and fp32). The GPU had no room for a second copy, so this is the noise
    floor: the same model, different arithmetic.
- Each candidate is ranked exactly as `gallery.search.Searcher` does (raw, k = 12) and compared with the server on:
  - top-12 overlap, and whether the top result is the same;
  - Kendall τ over the server's 12 similarities;
  - the largest similarity difference on those 12;
  - the weak flag.
- "cos vs torch bf16" is the cosine to the CPU bf16 vector.

## Results

#### gallery: 246 queries (216 on-topic, 30 nothing-close), 5558 items, weak_threshold 0.6677
| metric | torch_cpu_bfloat16 | onnx_fp32 | onnx_q8 | onnx_q4 | browser_q8_webgpu | browser_fp16_webgpu | browser_fp16_wasm | browser_q4_webgpu |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overlap@12 mean | 0.9922 | 0.9929 | 0.9892 | 0.8936 | 0.9892 | 0.9892 | 0.9929 | 0.8936 |
| overlap@12 min | 0.9167 | 0.9167 | 0.9167 | 0.5833 | 0.9167 | 0.8333 | 0.9167 | 0.5833 |
| queries with 12/12 | 0.9065 | 0.9146 | 0.8699 | 0.2561 | 0.8699 | 0.8740 | 0.9146 | 0.2561 |
| top-1 same | 0.9837 | 0.9797 | 0.9756 | 0.8496 | 0.9756 | 0.9756 | 0.9797 | 0.8496 |
| kendall tau mean | 0.9818 | 0.9828 | 0.9777 | 0.7885 | 0.9777 | 0.9784 | 0.9823 | 0.7885 |
| max |Δsim| median | 0.0006 | 0.0006 | 0.0008 | 0.0069 | 0.0008 | 0.0008 | 0.0006 | 0.0069 |
| max |Δsim| max | 0.0018 | 0.0018 | 0.0022 | 0.0218 | 0.0022 | 0.0023 | 0.0018 | 0.0218 |
| weak flag same | 1.0000 | 1.0000 | 1.0000 | 0.9919 | 1.0000 | 1.0000 | 1.0000 | 0.9919 |
| weak: on-topic flagged | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| weak: nothing-close flagged | 29 | 29 | 29 | 29 | 29 | 29 | 29 | 29 |
| server: on-topic flagged | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| server: nothing-close flagged | 29 | 29 | 29 | 29 | 29 | 29 | 29 | 29 |
| cos vs torch bf16 mean | 1.0000 | 1.0000 | 0.9999 | 0.9920 | 0.9999 | 1.0000 | 1.0000 | 0.9920 |
| cos vs torch bf16 min | 1.0000 | 0.9999 | 0.9999 | 0.9790 | 0.9999 | 0.9999 | 0.9999 | 0.9790 |
#### moon: 35 queries (25 on-topic, 10 nothing-close), 1024 items, weak_threshold 0.6697
| metric | torch_cpu_bfloat16 | onnx_fp32 | onnx_q8 | onnx_q4 | browser_q8_webgpu | browser_fp16_webgpu | browser_fp16_wasm | browser_q4_webgpu |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overlap@12 mean | 0.9952 | 0.9881 | 0.9905 | 0.8548 | 0.9905 | 0.9857 | 0.9881 | 0.8548 |
| overlap@12 min | 0.9167 | 0.9167 | 0.9167 | 0.6667 | 0.9167 | 0.9167 | 0.9167 | 0.6667 |
| queries with 12/12 | 0.9429 | 0.8571 | 0.8857 | 0.1429 | 0.8857 | 0.8286 | 0.8571 | 0.1429 |
| top-1 same | 1.0000 | 1.0000 | 1.0000 | 0.8000 | 1.0000 | 1.0000 | 1.0000 | 0.8000 |
| kendall tau mean | 0.9775 | 0.9784 | 0.9749 | 0.7481 | 0.9749 | 0.9758 | 0.9784 | 0.7481 |
| max |Δsim| median | 0.0007 | 0.0007 | 0.0008 | 0.0078 | 0.0008 | 0.0006 | 0.0007 | 0.0078 |
| max |Δsim| max | 0.0013 | 0.0023 | 0.0022 | 0.0160 | 0.0022 | 0.0025 | 0.0022 | 0.0160 |
| weak flag same | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| weak: on-topic flagged | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| weak: nothing-close flagged | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 10 |
| server: on-topic flagged | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| server: nothing-close flagged | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 10 |
| cos vs torch bf16 mean | 1.0000 | 1.0000 | 0.9999 | 0.9899 | 0.9999 | 0.9999 | 1.0000 | 0.9899 |
| cos vs torch bf16 min | 1.0000 | 1.0000 | 0.9999 | 0.9850 | 0.9999 | 0.9999 | 1.0000 | 0.9850 |

## What the browser can run

| Runtime | q4 / q4f16 / q8 | fp16 |
|---|---|---|
| WASM (no GPU) | does not load: `GatherBlockQuantized` kernel missing | loads |
| WebGPU (Edge, RTX 3070 Ti) | loads | loads |

Node's onnxruntime has that kernel, so Node results alone do not show what a browser can run.

Timings, measured once on this machine (Mason's PC):

| Path | Load, cold cache | Per query (median) |
|---|---|---|
| Edge, WebGPU, q8 | 18 to 26 s | 37 ms |
| Edge, WASM, fp16 | (not timed cold) | 275 ms |

On a second visit with a real browser profile the model loads from the browser's cache: 3.9 s, no download. A
fresh Playwright context is incognito-like, has a small storage quota, and does not keep the 314 MB weights file,
so a test there downloads every time.

## Decision

- **WebGPU:** q8, 346 MB with the tokenizer.
- **Without WebGPU:** fp16 on WASM, 575 MB.

Both sit at the noise floor:
- top-1 agreement is 97.6 to 98.0% (gallery) against 98.4% for the server's own code on CPU;
- the weak flag agrees on all 281 queries;
- weak_threshold separation is unchanged: 29 of 30 nothing-close and 1 of 216 on-topic flagged in the gallery,
  10 of 10 and 0 of 25 on the Moon, the same as the server.

q4 is rejected: top-1 85% (gallery) and 80% (Moon), and 12 of 12 overlap on only 26% and 14% of queries.

The site also leaves out the 182 AIC audio guides, as the public data release does. A query whose server answer
included one shows an 11th work in its place.

## Reproduce

Scripts are in `test_scripts/static_parity/` (`npm install` there first). Run from the repo root:

```bash
uv run python test_scripts/static_parity/server_results.py        # needs the server on :8000
(cd test_scripts/static_parity && node embed_onnx.mjs q8)          # likewise q4, q4f16, fp16, fp32
uv run python test_scripts/static_parity/embed_torch.py bfloat16   # and float32
# in a browser: test_scripts/static_parity/browser/ (README there)
uv run python test_scripts/static_parity/analyze.py
```
