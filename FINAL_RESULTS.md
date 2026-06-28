# FINAL RESULTS — the simple answer that works

## What ships
**`large-v3-turbo` fine-tuned (EL/ES/FR/EN + Common Voice EL) → Q3_K-matched QAT → Q3_K (368 MB)
→ Go runtime with beam search.** No PyTorch at runtime. CPU + GPU hybrid. Cross-platform.
The model is also continued on Common Voice Greek for broader real-world robustness.

> **368 MB floor:** Greek Q3_K bottoms out at ~0.148 because whisper.cpp Q3_K crushes the 253 MB
> token embedding to 3-bit (27 MB). More data/QAT improves the linears but can't break this at
> 368 MB. For lower WER use a bigger quant (the embedding survives): **Q4_K 474 MB → 0.124**,
> **Q5_K 574 MB → 0.110**. Just change `-model` to a Q4_K/Q5_K file.

## The two insights that got us to 368 MB @ 0.148
1. **Decode**: the earlier "0.285" was a **greedy-decode artifact** (repetition loops).
   **Beam search (size 5)** alone takes plain fine-tuned Q3_K to **0.197**.
2. **Q3_K-matched QAT**: train with the **EXACT ggml Q3_K quantizer in the loop** (straight-through
   estimator, via a ctypes call to ggml). Because training == deployment, the exported standard
   Q3_K GGUF keeps the quality: **0.148** (verified deployed == in-training, no mismatch).

```
plain fine-tuned Q3_K + greedy        ->  0.285   (bad)
plain fine-tuned Q3_K + beam 5         ->  0.197
Q3_K-matched QAT + beam 5  (THIS)      ->  0.148   <-- ships
baked INT3 then Q3_K (double-quant)    ->  0.355   (never do this)
```

## Greek WER — held-out FLEURS (real speech)
| Model | Size | Greek WER | Runtime |
|---|---|---|---|
| **Q3_K-matched QAT (this default)** | **368 MB** | **0.148** | Go, no PyTorch |
| plain fine-tuned Q3_K + beam | 368 MB | 0.197 | Go |
| fine-tuned Q4_K + beam | 474 MB | 0.124 | Go |
| fine-tuned Q5_K + beam | 574 MB | 0.110 | Go |
| FP16 fine-tuned (ceiling) | 1.6 GB | 0.108 | PyTorch |
| plain whisper.cpp Q3_K + greedy | 368 MB | 0.285 | Go |

ES/FR/EN stay near-perfect (~0.05–0.07) across all the above.

## Prompt tests (368 MB Q3_K, beam 5, Go app)
**Greek** — content correct; errors mostly on foreign proper names:
```
WER 0.04  Υπάρχει μεγάλος αριθμός εστιατορίων που περιβάλλουν τον κήπο...        ✅
WER 0.12  Σε αντίθεση με άλλα πρωτεύοντα, οι ανθρωπίδες δεν χρησιμοποιούν πια... ✅
```
**Spanish / French / English** — near-perfect:
```
ES WER 0.00  El ganador olímpico de la medalla de oro debía nadar en el estilo libre...
FR WER 0.00  Lorsque vous appelez une personne qui se trouve à des milliers de kilomètres...
EN WER 0.00  Members of a subculture often signal their membership through...
```

## Run it
```bash
# one clip
./whisperhybrid -profile quality -model model/ggml-large-v3-turbo-q3_k.bin -lang el clip.wav
# HTTP server (CPU+GPU hybrid)
./whisperhybrid -profile server -model model/ggml-large-v3-turbo-q3_k.bin -serve :8080 -lang el
```
`-profile quality` uses beam 5 (the setting that matters). Do **not** use greedy for production.

## Why the earlier complex paths are NOT used
- **Custom INT3 kernel (PyTorch, 0.176)**: best quality but PyTorch-only and ~690 MB. Kept as R&D.
- **Baked Q3_K (double-quant)**: 0.355 — destroys quality. Never use.
- **Q3_K-matched QAT**: would squeeze ~0.18–0.19, but plain Q3_K + beam already meets the goal,
  so it is optional R&D (see `docs/Q3K_MATCHED_QAT.md`).

## Bottom line
**368 MB, fast, Go, no PyTorch, Greek ~0.20 / ES-FR-EN near-perfect** — using the stock
whisper.cpp Q3_K kernels with beam search. Simple and shippable.
