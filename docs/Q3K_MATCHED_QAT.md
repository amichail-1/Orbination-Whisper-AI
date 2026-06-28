# Q3_K-matched QAT plan

## Objective

Produce a standard whisper.cpp `q3_k` model that keeps as much of the QAT/fine-tune quality as possible while remaining compatible with existing whisper.cpp CPU/GPU kernels.

The project already showed that generic group-16 INT3 QAT does not survive official `q3_k` re-quantization. The training quantizer must match the deployment quantizer.

## What “match Q3_K” means

The fake quantizer must reproduce the same dequantized float values as whisper.cpp’s Q3_K path for every quantized tensor, including:

- the 256-value super-block structure,
- 16 groups of 16 values,
- per-group scales that are themselves quantized,
- signed 3-bit values in the effective range used by Q3_K,
- the same packing/depacking math for `hmask`, `qs`, `scales`, and block scale `d`,
- the same rounding/clamping behavior as the C reference implementation.

Do not train with “close enough” group-wise INT3. It must be bit/float-equivalent after dequantization.

## Engineering steps

### 1. Pin the deployment source

Pick a specific `whisper.cpp` commit and record it in the model card/release notes.

```bash
git -C external/whisper.cpp rev-parse HEAD > MODELSOURCE.txt
```

The fake quantizer and exporter must be tested against this commit.

### 2. Build a Q3_K reference harness

Create a tiny C/C++ test binary that:

1. accepts a float tensor row,
2. calls whisper.cpp/ggml Q3_K quantization,
3. immediately dequantizes it,
4. writes the dequantized float row.

This gives the ground truth for the PyTorch fake quantizer.

### 3. Implement PyTorch STE fake quant

Forward pass:

```text
y = dequant_q3k(pack_q3k(x))
```

Backward pass:

```text
dL/dx = dL/dy
```

This is the straight-through estimator. The forward pass must use the exact Q3_K-equivalent dequantized values.

### 4. Validate equivalence before training

For random tensors and real model tensors:

```text
max_abs(fake_q3k_torch(x) - c_reference_q3k_dequant(x)) <= small tolerance
```

Run this per tensor shape and per layer family.

### 5. Train from the FP16 fine-tuned checkpoint

Use the best FP16 Greek/multilingual fine-tuned model as the starting point.

Suggested loss mix:

```text
loss = ASR supervised CE + teacher KD CE + optional encoder hidden-state distillation
```

Important settings:

- evaluate free-running WER regularly,
- keep beam-search eval in the loop,
- watch for repetition loops, not only loss,
- consider protecting the 4-layer decoder more than the encoder if Q3_K everywhere collapses,
- keep `q/k` protection experiments, but remember that a non-standard mixed type may change model size.

### 6. Export using the official quantizer

After QAT, export the FP16 weights and run the pinned whisper.cpp quantizer:

```bash
./build/bin/quantize qat-fp16.bin qat-q3_k.bin q3_k
```

Then compare:

```text
PyTorch fake-Q3_K WER vs exported whisper.cpp Q3_K WER
```

A large gap means the fake quantizer still does not match deployment.

## Acceptance criteria

Minimum useful result:

```text
Greek held-out FLEURS WER <= 0.20
Model size close to standard Q3_K size
No catastrophic repetition loops
Go runtime runs unchanged with -model qat-q3_k.bin
```

Strong result:

```text
Greek WER 0.17–0.18 at standard Q3_K size
```

Based on the project notes, the strong result may be near the practical limit for turbo’s 4-layer decoder. Treat `<=0.20` as the realistic first milestone.
