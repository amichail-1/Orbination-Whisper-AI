<div align="center">

# Orbination Whisper AI

**Quantization-aware compression of Whisper large-v3-turbo to a compact 368 MB, multilingual,
CPU/GPU speech-to-text engine — deployable as a single Go binary, with no Python at runtime.**

![Model](https://img.shields.io/badge/model-368_MB_·_Q3__K-1f6feb)
![Runtime](https://img.shields.io/badge/runtime-Go_+_whisper.cpp-00ADD8)
![Backends](https://img.shields.io/badge/backends-CPU_·_CUDA_·_Metal_·_Vulkan-555)
![Deps](https://img.shields.io/badge/runtime_deps-none_(no_PyTorch)-2ea043)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

</div>

---

## Abstract

`whisper-large-v3-turbo` delivers strong multilingual ASR but is impractical for on-device or
cost-sensitive deployment at 1.6 GB FP16. We compress it to **368 MB (3-bit, Q3_K)** while
preserving accuracy through **Q3_K-matched quantization-aware training (QAT)**: the network is
trained with the *exact* `ggml` Q3_K quantizer in the forward pass (straight-through gradients) plus
teacher distillation, so the exported standard Q3_K checkpoint deploys at the trained error rate with
no train/inference gap. The model is served by a portable **Go** runtime over `whisper.cpp` with
**hybrid CPU+GPU** scheduling and beam-search decoding.

---

## Key results

Word Error Rate on held-out **FLEURS** (real human speech), beam search, deployed Go runtime.

| Configuration | Size | English | Spanish | French | Greek |
|---|---|---:|---:|---:|---:|
| **Orbination Q3_K (default)** | **368 MB** | **0.065** | **0.050** | **0.065** | **0.148** |
| Orbination Q4_K | 474 MB | 0.062 | 0.048 | 0.063 | 0.124 |
| Orbination Q5_K | 574 MB | 0.061 | 0.047 | 0.061 | 0.110 |
| Fine-tuned FP16 (upper bound) | 1.6 GB | 0.061 | 0.046 | 0.060 | 0.108 |

High-resource languages stay essentially flat across precisions; the custom kernel's largest gains
appear on quantization-sensitive content (Greek: **0.285 → 0.148** at equal size — see *Method*).

Prebuilt binaries and the model are attached to each [Release](../../releases).

---

## Method

### Problem
Whisper-turbo has a shallow **4-layer decoder** (vs. a 32-layer encoder). Below ~4-bit, decoder
quantization error compounds autoregressively into degenerate output, and naive
fine-tune-then-quantize loses most of the fine-tuning gain to quantization noise. Post-training Q3_K
of the stock model, with greedy decoding, reaches only **0.285** WER on the hardest content.

### Q3_K-matched QAT
We remove the train/inference mismatch by placing the production quantizer **inside training**:

```
forward :  W_q = ggml_Q3_K_dequant( ggml_Q3_K_quant( W ) )      # exact ggml kernel (via ctypes)
backward:  ∂L/∂W = ∂L/∂W_q                                       # straight-through estimator
loss    :  CE(student) + KD( student ‖ FP16 teacher )           # teacher distillation
```

1. A C shim exposes ggml's real Q3_K quantize/dequantize; in-training values are bit-identical to
   what `whisper.cpp` computes at inference.
2. Every encoder and decoder linear is wrapped with this exact round-trip; gradients flow via a
   straight-through estimator (a periodic-refresh cache + OpenMP keep training tractable).
3. The quantized student is distilled against the FP16 teacher's logits.
4. The trained weights are exported and quantized with the *same* Q3_K — so the deployed WER equals
   the in-training WER (verified, not extrapolated).

A second factor is decode-time: **beam search (size 5)** with temperature fallback removes the
repetition loops that inflate greedy WER.

| Stage (hardest-content WER) | |
|---|---|
| Stock Q3_K, greedy decode | 0.285 |
| Fine-tuned Q3_K, beam search | 0.197 |
| **Q3_K-matched QAT, beam search** | **0.148** |
| Bake-INT3-then-Q3_K (double quantization) | 0.355 — *anti-pattern* |

### The 368 MB floor
At 368 MB, accuracy is bounded by the token-embedding quantization: `whisper.cpp` Q3_K compresses
the 253 MB embedding to 3-bit (27 MB), which dominates the residual error. Since whisper.cpp uses a
single per-file precision, additional QAT improves the linear layers but cannot break this floor at
368 MB; giving the embedding more bits (Q4_K/Q5_K) is what lowers it further.

---

## System

A single Go application wraps `whisper.cpp` via CGo and adds:

- **Hybrid CPU+GPU scheduling** — GPU and CPU contexts in one process; light load runs on GPU,
  bursts fan out to both via work-stealing, with a load-aware CPU "assist" threshold.
- **Beam-search decoding** (size 5) with temperature fallback.
- **Runtime profiles** (`quality` / `balanced` / `speed` / `tiny` / `server`) and `-model auto`.
- **CLI and HTTP server** (`/inference`, `/stats`, `/health`).
- **Build-tag backends**: `cpu` (default) · `cuda` · `metal` · `vulkan` · `rocm`.

Portability is inherited from `whisper.cpp`/ggml: AVX/AVX-512 (x86), NEON (ARM64), CUDA (NVIDIA),
Metal (Apple), Vulkan (others). The Go sources are platform-independent; only the native library is
rebuilt per target.

---

## Getting started

Prebuilt binaries for Windows, macOS (Apple Silicon) and Linux are on the
[Releases](../../releases) page, alongside the model.

```bash
# from source:
# 1. download the model from Releases into app/model/
# 2. build whisper.cpp for your platform (app/BUILD.md), copy libs into app/lib/
cd app
go build -tags cuda -o whisperhybrid .          # cuda | metal | vulkan | rocm | (none = CPU)
./whisperhybrid -profile quality -model auto -lang en clip.wav
./whisperhybrid -profile server  -model auto -serve :8080            # HTTP API
```
`-lang` accepts a language code or `auto`.

---

## Reproducibility

```
training/q3k_qat.py     Q3_K-matched QAT (exact ggml Q3_K + STE + teacher KD)
training/quant_lib.c    C shim exposing the ggml Q3_K round-trip
converter/              pure-Go GGUF baker (no PyTorch)
scripts/                whisper.cpp build + quantization
tools/                  WER evaluation and profile sweeps
```

Pipeline: FP16 fine-tune → Q3_K-matched QAT → bake to FP16 GGUF → `whisper-quantize q3_k` → 368 MB.

---

## Repository structure

```
app/         Go runtime (engine, dispatcher, profiles, multi-backend link tags) + BUILD.md
converter/   pure-Go GGUF baker
training/    Q3_K-matched QAT
scripts/     build & quantization
tools/       evaluation
.github/     CI: prebuilt binaries for Windows / macOS / Linux (CPU & GPU)
```

---

## Citation

```bibtex
@software{orbination_whisper_ai,
  title        = {Orbination Whisper AI: Q3_K-matched QAT compression of Whisper-large-v3-turbo},
  author       = {Michail, Antonios},
  organization = {Leia Enterprise Solutions},
  year         = {2026},
  url          = {https://github.com/amichail-1/Orbination-Whisper-AI},
  note         = {www.leia.gr, www.orbination.com}
}
```

## Acknowledgements
Built on [`openai/whisper`](https://github.com/openai/whisper) and
[`ggerganov/whisper.cpp`](https://github.com/ggerganov/whisper.cpp); evaluated on
[FLEURS](https://huggingface.co/datasets/google/fleurs).

## License
MIT © 2026 Leia Enterprise Solutions.

---

<div align="center">

**Orbination Whisper AI** — an [Orbination](https://www.orbination.com) application
<br>© 2026 [Leia Enterprise Solutions](https://www.leia.gr) · [www.leia.gr](https://www.leia.gr) · [www.orbination.com](https://www.orbination.com)

</div>
