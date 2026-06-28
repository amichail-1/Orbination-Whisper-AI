# Whisper Turbo Go — smallest/fastest/best-quality practical build

This is a Go + `whisper.cpp` runtime for local/offline Whisper transcription on macOS, Linux, Windows, and ARM64 targets.

The project goal is aggressive but realistic:

```text
smallest possible + fastest possible + best quality
```

Those three goals conflict. The source now handles this with runtime **profiles** and one clear default.

## The decision

Ship a **single model**: `large-v3-turbo` fine-tuned (EL/ES/FR/EN) → **Q3_K (368 MB)**,
decoded with **beam search**. See `FINAL_RESULTS.md` for the full story.

| Target | Approx size | Greek WER | Role |
|---|---:|---:|---|
| **fine-tuned Q3_K + beam (THIS)** | **368 MB** | **0.197** | **The shipping model.** |
| fine-tuned Q3_K + greedy | 368 MB | 0.285 | Don't use greedy. |
| Q3_K-baked double-quant | 368 MB | 0.355 | Do not use. |
| FP16 fine-tuned | 1.6 GB | 0.110 | Quality ceiling, not small. |

Key point: the 368 MB Q3_K was always good — **beam search** (not greedy) makes it 0.197.

```text
Ship:          model/ggml-large-v3-turbo-q3_k.bin + beam search (profile quality)
R&D frontier:  Q3_K-matched QAT for ~0.18 (optional, see docs/Q3K_MATCHED_QAT.md)
```

## Runtime profiles

The app supports:

```bash
-profile quality   # default: baked Q4_K, beam 5
-profile balanced  # baked Q4_K, beam 3
-profile speed     # Q3_K preferred, greedy decode
-profile tiny      # Q3_K preferred, smallest package
-profile server    # Q4_K preferred, GPU first + CPU assist
```

Recommended commands:

```bash
# Best shippable quality per MB
./whisperhybrid -profile quality -model auto -lang el clip.wav

# Smallest / fastest
./whisperhybrid -profile tiny -model auto -lang el clip.wav

# HTTP server
./whisperhybrid -profile server -model auto -serve :8080 -lang el
```

`-model auto` searches `./model`, `./app/model`, and a `model` folder next to the executable. It prefers files named like:

```text
baked-q4_k.bin
baked-q4_k.gguf
ggml-large-v3-turbo-q4_k.bin
ggml-large-v3-turbo-q3_k.bin
```

You can always force a model:

```bash
./whisperhybrid -profile quality -model app/model/baked-q4_k.bin -lang el clip.wav
./whisperhybrid -profile tiny -model app/model/ggml-large-v3-turbo-q3_k.bin -lang el clip.wav
```

## Build whisper.cpp libraries

Unix examples:

```bash
scripts/build_whispercpp_unix.sh cpu
scripts/build_whispercpp_unix.sh metal
scripts/build_whispercpp_unix.sh cuda
scripts/build_whispercpp_unix.sh vulkan
```

Windows PowerShell:

```powershell
scripts\build_windows.ps1 -Backend cpu
scripts\build_windows.ps1 -Backend cuda -Tags cuda
scripts\build_windows.ps1 -Backend vulkan -Tags vulkan
```

## Build the Go app

```bash
scripts/build_app_unix.sh whisperhybrid              # CPU
scripts/build_app_unix.sh whisperhybrid -tags metal  # macOS Metal
scripts/build_app_unix.sh whisperhybrid -tags cuda   # Linux NVIDIA CUDA
scripts/build_app_unix.sh whisperhybrid -tags vulkan # Vulkan
scripts/build_app_unix.sh whisperhybrid -tags rocm   # Linux ROCm/HIP
```

The Go code is portable, but the native CGo libraries are not. Build one artifact per OS/architecture/backend.

## Build an optimal release folder

```bash
scripts/build_optimal_release_unix.sh metal models/baked-fp16.bin dist/macos-arm64-metal
scripts/build_optimal_release_unix.sh cuda  models/baked-fp16.bin dist/linux-amd64-cuda
scripts/build_optimal_release_unix.sh cpu   models/baked-fp16.bin dist/linux-amd64-cpu
```

The release folder contains:

```text
whisperhybrid
model/baked-q4_k.bin        # default quality model
model/*q3_k*.bin            # optional tiny/fast model, if available
lib/                        # native whisper.cpp libraries
run-quality.sh
run-tiny.sh
```

## Optional embedded-model executable

A true single executable is possible per platform/backend, but the binary becomes hundreds of MB and model updates require rebuilding the executable.

```bash
cp /path/to/baked-q4_k.bin app/model/
cd app
CGO_ENABLED=1 go build -tags "embedded_model" -trimpath -ldflags="-s -w" -o whisperhybrid .
./whisperhybrid -model :embedded -profile quality -lang el clip.wav
```

For production, a folder bundle is usually easier to update.

## Quantize models

Plain whisper.cpp quantization:

```bash
./build/bin/quantize models/ggml-large-v3-turbo.bin models/ggml-large-v3-turbo-q3_k.bin q3_k
./build/bin/quantize models/ggml-large-v3-turbo.bin models/ggml-large-v3-turbo-q4_k.bin q4_k
```

Project-specific baked path:

```bash
cd converter
go run . ../models/base-fp16.bin ../models/qat-model.safetensors ../models/baked-fp16.bin
../external/whisper.cpp/build/bin/quantize ../models/baked-fp16.bin ../app/model/baked-q4_k.bin q4_k
```

Avoid:

```text
custom INT3 fake-quant bake -> official q3_k quantize
```

The project results show that double-quantization destroys the QAT benefit.

## Evaluate profiles

Create a CSV manifest:

```csv
audio,reference,language
/path/to/a.wav,"reference transcript",el
```

Run one profile:

```bash
python3 tools/wer_eval.py \
  --manifest fleurs-el-heldout.csv \
  --bin ./app/whisperhybrid \
  --model auto \
  --profile quality \
  --lang el \
  --out results-el-quality.csv
```

Compare profiles:

```bash
python3 tools/profile_sweep.py \
  --manifest fleurs-el-heldout.csv \
  --bin ./app/whisperhybrid \
  --models-dir ./app/model \
  --lang el \
  --out profile_sweep.csv
```

## Important constraints

- There is no one native CGo/GPU binary that runs unchanged across macOS, Linux, and Windows.
- A fully static GPU binary is usually unrealistic because CUDA/Vulkan/driver runtimes remain system dependencies.
- The current Go runtime can run standard `q3_k`/`q4_k`/`q5_k` models through whisper.cpp. The remaining quality frontier is model/training/quantizer alignment, not the Go wrapper.
