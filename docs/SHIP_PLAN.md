# Shipping plan

## Release target 1: practical high-quality release

Ship **Q4_K-baked** first.

```text
Artifact size: ~474 MB model + app/libs
Expected Greek WER: around 0.189 from project tests
Runtime: Go + whisper.cpp, no PyTorch
Decode: beam=5 for Greek quality
Backends: CPU, Metal, CUDA, Vulkan as separate builds
```

Why this is the best first release:

- It preserves most of the QAT benefit after deployment quantization.
- It uses existing whisper.cpp kernels, so speed is already good on CPU/GPU.
- It avoids custom C/CUDA/Metal kernel work.
- It avoids the proven bad path: custom INT3 bake followed by official Q3_K double quantization.

Recommended command:

```bash
./whisperhybrid -profile quality -model auto -lang el audio.wav
```

Server command:

```bash
./whisperhybrid -profile server -model auto -serve :8080 -lang el
```

## Release target 2: smallest standard release

Ship plain **Q3_K** as a separate “small/fast” model only.

```text
Artifact size: ~368 MB model + app/libs
Expected Greek WER: materially worse than Q4_K-baked in project tests
Runtime: existing whisper.cpp Q3_K kernels
```

Use this when download size or memory matters more than Greek quality.

Recommended command:

```bash
./whisperhybrid -profile tiny -model auto -lang el audio.wav
```

## Research target: 368 MB with good Greek quality

Implement **Q3_K-matched QAT**.

The key requirement is that the training fake quantizer must match the exact dequantized values produced by whisper.cpp’s Q3_K packing/dequantization. Generic group-16 INT3 is not enough.

Acceptance target:

```text
Model size: ~368 MB
Greek WER: <= 0.20 on held-out FLEURS
Runtime: unchanged Go app + standard whisper.cpp Q3_K kernels
No repetition loops under free-running beam search
```

## Research target: best quality small model

Implement a custom ggml quant type for the project’s group-16 INT3/outlier kernel.

This is only worth doing after Q3_K-matched QAT is proven insufficient, because it requires:

- new ggml type enum + type traits,
- new block layout,
- reference dequantization,
- CPU vector dot kernel,
- CUDA/Metal/Vulkan kernels for speed,
- converter support,
- maintenance across whisper.cpp/ggml updates.
