# Custom ggml INT3/outlier quant type plan

## Objective

Preserve the project’s custom group-16 INT3/INT4 + protected q/k + FP16 outlier behavior directly in ggml instead of baking it into FP16 and then re-quantizing.

This path targets maximum quality around the project’s custom-kernel result, but it is a larger runtime engineering project than Q3_K-matched QAT.

## Proposed type

Use a new internal type name such as:

```text
GGML_TYPE_QG3_16_O
```

Meaning:

```text
QG3: group-wise 3-bit signed quant
16:  group size 16
O:   explicit sparse/high-magnitude outlier support
```

The exact block design should be finalized after measuring alignment, vectorization, and outlier density.

## Required ggml changes

1. Add the type enum and type traits.
2. Add a block struct for packed INT3 values, group scales, and optional outlier metadata.
3. Add reference quantize/dequantize functions.
4. Add CPU `vec_dot`/matmul kernels.
5. Add CUDA kernel for NVIDIA.
6. Add Metal kernel for macOS Apple Silicon.
7. Add Vulkan later if cross-vendor GPU support matters.
8. Extend the converter so it writes the new tensor type directly.
9. Keep the Go wrapper unchanged except for linking against the patched whisper.cpp/ggml library.

## Main risk

Without real packed INT3 matmul kernels, the model either expands back to FP16 at runtime or runs slowly through dequantization. This path only pays off when the custom storage format and fast kernels both exist.

## When to choose this path

Choose this only if Q3_K-matched QAT cannot reach acceptable Greek WER at 368 MB.
