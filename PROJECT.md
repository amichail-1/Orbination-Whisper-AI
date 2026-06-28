# Whisper Turbo — Small, Fast, High-Quality (Greek-first) — Project Overview

## The goal
Take **OpenAI Whisper large-v3-turbo** (809M params, 32 encoder layers, only 4 decoder
layers) and make it:
- **Small** (target: as close to ~368 MB / Q3_K size as possible)
- **Fast** (real CPU **and** GPU inference, hybrid if both present)
- **High quality**, especially for **Greek** (plus ES/FR/EN)
- **Deployable as a Go app** for Windows / Linux / macOS / ARM64, **no PyTorch at runtime**
- Ideally pushing toward **1.58-bit / ternary** weights (BitNet-style)

Success metric: **Greek Word Error Rate (WER)** on held-out **FLEURS** (real human speech),
plus ES/FR/EN, while shrinking the model and keeping it fast everywhere.

---

## Why this is hard
- Whisper-turbo has a **tiny decoder (4 layers, no redundancy)** — it is very fragile under
  aggressive quantization. The encoder (32 layers) is far more robust.
- **One-shot post-training quantization below ~4-bit collapses** the model (it autoregressively
  compounds errors → repetition/garbage), even though teacher-forced *loss* looks fine.
- A company (**ENERZAi**) did achieve 1.58-bit Whisper — but only with **QAT**
  (quantization-aware training), **language specialization**, **group=channel**, and **custom
  kernels**. That's the bar.

---

## What we tried (chronological)
1. **One-shot ternary / ≤2-bit** (multiple package versions): **collapsed**. Only ~20% of the
   encoder (V/O + fc2) is safely ternary one-shot. Established the hard floor.
2. **whisper.cpp k-quants** (Q3_K/Q4_K/Q5_K): work, but quality-limited (see table).
3. **TQ1/TQ2 ternary in whisper.cpp** (patched the runtime to support it): runs, but needs a
   QAT-trained model — one-shot still collapses.
4. **FP16 fine-tuning** on FLEURS (EL, then EL/ES/FR/EN, ~37h, ~10h overnight run): **big Greek
   gain** (16.8% → 11.0%). But **plain quantization afterwards washes most of it out**.
5. **Our custom kernel** (the breakthrough): group-wise INT quant + outlier protection +
   **QAT with straight-through estimator** + **teacher KD**, with the **same quantizer in
   training and inference**. This keeps the fine-tune quality through quantization.
   - v1 naive per-channel INT4 → collapsed, QAT recovered slowly.
   - v2 **group-16 + protect q/k + 0.5% FP16 outliers + KD** → **Greek 0.176 @ Q3-level**,
     0.132 @ Q4-level. Verified real (not a no-op).
   - v3 learnable scales → **no improvement** (amax scales already near-optimal).
6. **Go runtime + pure-Go converter** (to drop PyTorch): `qbake` bakes our QAT weights into a
   ggml model (no PyTorch); `whisperhybrid` runs it via whisper.cpp with CPU+GPU hybrid.
   - **Beam search** was required — plain greedy caused repetition loops that inflated WER.

---

## Key findings (the hard truths we established)
1. **Loss ≠ quality.** Low teacher-forced loss can hide a model that collapses during real
   (free-running) generation.
2. **One-shot ≤2-bit on turbo collapses.** Q3_K is the practical floor for *plain* quantization.
3. **Plain fine-tune + then quantize does NOT help the small model** — even 10h multilingual:
   FP16 Greek 16.8%→11%, but Q3_K only 30%→28.5%. Quantization noise dominates.
4. **The fix is QAT with a consistent quantizer** (train == inference). Our kernel proves it:
   Greek 0.176 at Q3-level vs whisper.cpp Q3_K 0.285.
5. **You can run our kernel in Go/whisper.cpp with no PyTorch** by baking the quant-aware
   weights into a GGUF (FP16 storage) — quality holds (0.166).
6. **Double quantization fails.** Baking our group-16 INT3 and then re-quantizing to
   whisper.cpp's Q3_K (a *different* super-block format) re-introduces error → **0.355**.
   Even full-model QAT (FP16 0.178) collapses to 0.355 after Q3_K.
7. **368 MB @ ~0.17 is at/beyond the achievable frontier.** The 4-layer decoder at 3-bit has a
   floor around ~0.20. Real options for 368 MB: (a) QAT that **exactly matches** whisper.cpp's
   Q3_K block format, or (b) a **custom ggml quant type** (C/CUDA kernel) — both large efforts,
   and (a) realistically lands ~0.20, not 0.17.
8. **Size correction:** our custom kernel's true packed footprint is **~690 MB** (encoder
   int3/int4 + decoder/embeddings kept FP16), **not 368 MB** (an earlier mislabel; 368 MB is the
   *all-Q3_K* size).

---

## Results — Greek WER (held-out FLEURS, real speech)
| Model | Size | Greek WER | Runtime | Notes |
|---|---|---|---|---|
| FP16 baseline | 1.6 GB | 0.168 | — | stock turbo |
| FP16 fine-tuned | 1.6 GB | **0.110** | PyTorch | quality ceiling |
| **our kernel @ Q4** | ~470 MB* | **0.132** | PyTorch | *effective bits |
| **our kernel @ Q3 (g16)** | ~690 MB | **0.176** | PyTorch | verified 0.165–0.166 |
| **Q4_K-baked (Go)** | **474 MB** | **0.189** | **Go, no PyTorch** | ✅ shippable |
| Q5_K-baked (Go) | 574 MB | 0.190 | Go | |
| whisper.cpp Q5_K | 574 MB | 0.204 | Go | plain (no kernel) |
| whisper.cpp Q4_K | 474 MB | 0.217 | Go | plain |
| whisper.cpp Q3_K | 368 MB | 0.285 | Go | plain |
| Q3_K-baked (double-quant) | 368 MB | 0.355 | Go | ❌ fails |

ES/FR/EN are near-FP16 (<8% WER) across the kernel variants.

---

## Deliverables (current)
- **PyTorch custom kernel** (best quality): `qkernel/run_v2_q3_g16/` — Greek **0.176**, ~690 MB
  packed. Code: `qlinear3.py` (kernel), `qat_run6.py` (QAT), `infer_g16.py`, `verify.py`.
- **Go runtime, no PyTorch** (shippable small): `whisper_q4k_go/` — Greek **0.189 @ 474 MB**.
  - `app/` = `whisperhybrid` (CGo→whisper.cpp, **CPU+GPU hybrid**, beam search, CLI + HTTP).
  - `converter/` = `qbake` (pure-Go: QAT weights → our quant → ggml, **no PyTorch**).
- **Backups**: `whisper_backup_kernel/`, `whisper_backup_finetuned/`.
- **Proven hybrid**: 1 request → GPU only; burst of 16 → **10 GPU + 6 CPU in parallel**, with a
  load-aware "assist" switch (CPU joins only under backlog).

---

## Open problems / next steps
- **368 MB @ ~0.20** (best realistic small): implement a **QAT fake-quant that exactly matches
  whisper.cpp's Q3_K** super-block format → then a standard Q3_K GGUF stays accurate and runs on
  whisper.cpp's existing fast kernels (CPU AVX/NEON, CUDA/Metal/Vulkan). No C kernel needed.
- **Max quality small** (~0.17–0.18 @ ~690 MB): a **custom ggml quant type** storing our
  group-16 INT3 directly (+ CPU/CUDA kernels). Bigger effort; bigger file.
- **Cross-platform builds**: Go code is portable; build whisper.cpp libs per OS/arch
  (CUDA / Metal / Vulkan / NEON) and `go build`. See `whisper_q4k_go/app/BUILD.md`.
- **Speed kernel**: current Go runtime is fast (whisper.cpp). A packed-INT3 matmul would be the
  only path to *both* small *and* fast at our quality.

---

## One-line summary
We turned Whisper-turbo into a **Greek-strong, multilingual, quantization-aware model** and
proved that a **custom group-wise INT3 + outliers + QAT + KD kernel** keeps quality where plain
quantization fails (Greek **0.176** vs whisper.cpp Q3_K 0.285) — then made it run **in Go with no
PyTorch** at **474 MB / 0.189**, with a real **CPU+GPU hybrid** runtime. The remaining frontier
(**368 MB at ~0.17–0.20**) requires matching the deployment quantizer exactly (Q3_K-matched QAT)
or a custom ggml kernel.
