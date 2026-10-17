# Phase 0 — model feasibility (measured 2026-10-15)

Machine: WSL2, RTX 3070 Ti 8 GB. Windows desktop apps hold part of the VRAM outside WSL; `nvidia-smi`
showed between 0.7 GB and 5.5 GB used by them at different moments today, so VRAM headroom varies.
Raw numbers: `reports/phase0_embed.json`, `reports/phase0_gemma.json`.

## EmbeddingGemma 2 (`google/embeddinggemma-2`)

- **Loading path:** Hugging Face, sentence-transformers 6.1.0 + transformers 5.20, `SentenceTransformer("google/embeddinggemma-2", model_kwargs={"torch_dtype": torch.bfloat16}, config_kwargs={"audio_config": None})` on `cuda`. Not gated (no token needed). Not Ollama.
- **dtype:** bfloat16. The model card says float16 gives NaNs or degraded embeddings, so `Embedder` refuses it.
- **Prompts (from the model card / `config_sentence_transformers.json`):**
  - query: `task: search result | query: {query}` (prompt name `query` / `SearchQuery`)
  - document: `title: {title} | text: {content}`; `title: none | text: {content}` when there is no title. Wiki items are embedded as `title: {article title} | text: {raw_text}`.
  - images: no prefix, input `{"image": path}`.
  - image+title (interleaved, supported by the model card): `{"text": "title: {title} | <|image|>", "image": path}`.
- **Cross-modal sanity (fixtures, cosine):**

  | query | galaxy.jpg | still_life.jpg |
  |---|---|---|
  | "a spiral galaxy" | 0.762 | 0.558 |
  | "a bowl of fruit" | 0.564 | 0.748 |

  With image+title interleaving: 0.798 / 0.596 and 0.602 / 0.724.
- **Single text query latency (50 runs, after warm-up):** p50 77.9 ms, p95 87.8 ms.
- **Peak VRAM** (`torch.cuda.max_memory_allocated`): 849 MiB after load; 3,458 MiB peak in the benchmark (image batch 32); 2,716 MiB peak during the full corpus run (image batch 8). Peak reserved in the benchmark: 4,814 MiB.
- **Image throughput (64 images):** batch 8: 11.2 img/s, batch 16: 7.2 img/s, batch 32: 4.6 img/s. Larger batches were slower here (likely VRAM pressure from the Windows desktop at the time), so the default image batch is 8.
- **Text document throughput:** 270 texts/s (batch 32, ~70-word texts).
- **Load time:** 128–175 s cold on this machine, mostly Python imports from `/mnt/o` (9p) plus HF Hub metadata calls. The weights themselves load in < 1 s.
- **Full corpus (`gallery.embed_corpus`):** 5,100 items (3,400 images, 1,700 text) in 318 s of embedding (16.0 rows/s overall; images 8–12/s, text 70–190/s). Output passes `tests/test_vectors.py`.

## Gemma 4 E4B (llama.cpp, docker)

- **Image:** `ghcr.io/ggml-org/llama.cpp:server-cuda` (build `b11669-620d2225c`).
- **Weights:** `ggml-org/gemma-4-E4B-it-GGUF` → `gemma-4-E4B-it-Q4_0.gguf` (4.59 GB) + `mmproj-gemma-4-E4B-it-Q8_0.gguf` (0.56 GB), in `models/gemma-4-E4B-it-GGUF/` (gitignored). That repo has no Q4_K_M; Q4_0 is the ggml-org quant.
- **Run command:**

  ```bash
  docker run -d --name gallery-gemma4 --gpus all -p 8081:8080 \
    -v /mnt/o/Ode/Github/embeddings-gemma-exploration/models:/models \
    ghcr.io/ggml-org/llama.cpp:server-cuda \
    -m /models/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_0.gguf \
    --mmproj /models/gemma-4-E4B-it-GGUF/mmproj-gemma-4-E4B-it-Q8_0.gguf \
    --alias gemma-4-E4B-it -ngl 99 -c 8192 -np 4 --host 0.0.0.0 --port 8080 --jinja
  ```
  Stop/start later with `docker stop gallery-gemma4` / `docker start gallery-gemma4`. API: `http://localhost:8081/v1` (OpenAI-compatible); `/props` reports vision enabled, 4 slots, 2,048 ctx per slot.
- **VRAM:** total GPU use went from 2,764 → 6,545 MiB on the first start and 681 → 4,867 MiB on a later start, so the server takes about **3.8–4.2 GB**. The container also holds ~10 GB of host RAM.
- **Request:** `chat_template_kwargs: {"enable_thinking": false}`, temperature 0.4, max_tokens 300. The image is sent as a base64 data URL with title/creator/date/collection/description. The prompts are in `src/gallery/llm.py`. Output is accepted at 40–120 words, otherwise retried.
- **Seconds per image summary (mean over 10, sequential, fixtures):** **0.99 s** (range 0.75–1.28 s).
- **4 concurrent requests (12 image items):** 5.8 s total = 0.49 s per item.
- **Estimated curation time for 5,100 items (made before the run):** 5,100 × 0.49 s ≈ **42 min** with 4 workers (≈ 84 min sequential).
- **Measured on the real corpus:** 89 items/min with `--workers 4` (≈ 0.67 s/item), so ≈ **57 min** for 5,100 items. Started 23:03:04. 0 failures and 0 retries in the first 3,480 items.

## Example summaries (verbatim, from `data/summaries.jsonl`)

**aic:61603 — Edgar Degas, *Ballet at the Paris Opéra*, 1877 (image item)**
> This pastel and monotype by Edgar Degas, created in 1877, depicts a group of dancers on stage at the Paris Opéra. The view is from the orchestra pit, where the necks of double basses intrude into the scene. Degas employs his characteristic cropped forms and unusual vantage points to convey the immediacy of the moment. The delicate pastel medium captures the ephemeral beauty of the dancers, suggesting a rehearsal rather than a formal performance.

**nasa:PIA07902 — *Planetary Nebula NGC 7293 also Known as the Helix Nebula* (image item)**
> This ultraviolet image captures the planetary nebula NGC 7293, commonly known as the Helix Nebula. The structure showcases the dramatic end stages of a star similar to our Sun. As the star exhausts its fuel, it expels gas outward, forming this intricate shell of glowing material. This process ultimately leaves behind a much hotter, smaller, and denser white dwarf star.

**wiki:364788:0 — *Alfred Stieglitz* (text item)**
> Alfred Stieglitz, an American photographer and modern art promoter, dedicated his fifty-year career to establishing photography as a respected art form. Beyond his photographic work, he operated New York art galleries in the early twentieth century, serving as a conduit for introducing various avant-garde European artists to the American audience. Stieglitz was also married to the painter Georgia O'Keeffe.

## Notes

- The GPU never holds both models: `embed_corpus` ran with the Gemma container stopped, then the container was started for curation. The runtime app (EmbeddingGemma only, ~1–2.7 GB) should not run while curation is going on unless VRAM allows it.
- `SentenceTransformer` makes HF Hub metadata calls on load. To run fully offline (SPEC N4), set `HF_HUB_OFFLINE=1` when the app starts; the model is already in the local HF cache.
