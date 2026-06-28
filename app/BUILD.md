# Build notes: macOS, Linux, Windows

This app uses one Go codebase plus native `whisper.cpp` libraries. Build one release per OS/architecture/backend.

## Default product target

For the current project goal, use:

```text
Default quality artifact: baked Q4_K model + -profile quality
Optional tiny artifact:   Q3_K model + -profile tiny
```

The app defaults to:

```bash
./whisperhybrid -profile quality -model auto -lang el audio.wav
```

## 1. Choose your target backend

| Platform | Preferred backend | Notes |
|---|---|---|
| macOS Apple Silicon | Metal | Best local path for M-series Macs. |
| macOS Intel | CPU / Metal if supported | CPU may be simpler. |
| Linux NVIDIA | CUDA | Fastest for NVIDIA. |
| Linux AMD/Intel GPU | Vulkan or ROCm/HIP | Vulkan is easier; ROCm can be faster where supported. |
| Windows NVIDIA | CUDA | Build on Windows. |
| Windows AMD/Intel | Vulkan | Easier than vendor-specific paths. |
| Linux ARM64 | CPU NEON | Build on ARM64 or use a CGo-capable cross toolchain. |

## 2. Build whisper.cpp

From repo root:

```bash
scripts/build_whispercpp_unix.sh cpu
scripts/build_whispercpp_unix.sh metal
scripts/build_whispercpp_unix.sh cuda
scripts/build_whispercpp_unix.sh vulkan
```

Windows:

```powershell
scripts\build_windows.ps1 -Backend cpu
scripts\build_windows.ps1 -Backend cuda -Tags cuda
scripts\build_windows.ps1 -Backend vulkan -Tags vulkan
```

The scripts copy headers to `app/inc/` and native libs to `app/lib/`.

## 3. Backend build tags

| Tag | Link file | Use for |
|---|---|---|
| none | `link_cpu.go` | CPU build |
| `metal` | `link_metal.go` | macOS Metal |
| `cuda` | `link_cuda.go` | NVIDIA CUDA |
| `vulkan` | `link_vulkan.go` | Vulkan |
| `rocm` | `link_rocm.go` | Linux ROCm/HIP |

Examples:

```bash
cd app
CGO_ENABLED=1 go build -trimpath -ldflags="-s -w" -o whisperhybrid .
CGO_ENABLED=1 go build -tags metal -trimpath -ldflags="-s -w" -o whisperhybrid .
CGO_ENABLED=1 go build -tags cuda  -trimpath -ldflags="-s -w" -o whisperhybrid .
```

Windows:

```powershell
cd app
$env:CGO_ENABLED="1"
go build -trimpath -ldflags="-s -w" -o whisperhybrid.exe .
go build -tags cuda -trimpath -ldflags="-s -w" -o whisperhybrid.exe .
go build -tags vulkan -trimpath -ldflags="-s -w" -o whisperhybrid.exe .
```

## 4. Runtime profiles

| Profile | Model preference | Decode | Goal |
|---|---|---|---|
| `quality` | baked Q4_K | beam 5 | Best shippable quality per MB. |
| `balanced` | baked Q4_K | beam 3 | Faster, small quality tradeoff. |
| `speed` | Q3_K | greedy | Lowest latency. |
| `tiny` | Q3_K | greedy | Smallest package. |
| `server` | baked Q4_K | beam 5 | Throughput with GPU + CPU assist. |

Examples:

```bash
./whisperhybrid -profile quality -model auto -lang el clip.wav
./whisperhybrid -profile tiny -model auto -lang el clip.wav
./whisperhybrid -profile server -model auto -serve :8080 -lang el
```

## 5. Package

A practical portable package is:

```text
whisperhybrid(.exe)
lib/                native whisper.cpp libraries, if dynamic libs are used
model/              baked-q4_k.bin and optionally q3_k.bin
run-quality.sh      Unix helper to set library path and run quality profile
run-tiny.sh         Unix helper to set library path and run tiny profile
```

Build this folder on Unix-like systems:

```bash
scripts/build_optimal_release_unix.sh metal models/baked-fp16.bin dist/macos-arm64-metal
scripts/build_optimal_release_unix.sh cuda  models/baked-fp16.bin dist/linux-amd64-cuda
scripts/build_optimal_release_unix.sh cpu   models/baked-fp16.bin dist/linux-amd64-cpu
```

## 6. Embedded model option

For a real single executable, copy exactly one model into `app/model/` and build:

```bash
CGO_ENABLED=1 go build -tags "embedded_model" -trimpath -ldflags="-s -w" -o whisperhybrid .
CGO_ENABLED=1 go build -tags "embedded_model metal" -trimpath -ldflags="-s -w" -o whisperhybrid .
CGO_ENABLED=1 go build -tags "embedded_model cuda" -trimpath -ldflags="-s -w" -o whisperhybrid .
./whisperhybrid -model :embedded -profile quality clip.wav
```

Embedding is convenient but makes the executable hundreds of MB and makes model updates require a new binary.

## 7. Known limitations

- The bundled WAV reader supports PCM16 WAV only. Convert MP3/M4A/FLAC with ffmpeg before sending to the app.
- Multiple workers load multiple model contexts. This improves throughput only when the hardware has enough memory and parallel capacity.
- GPU builds usually still depend on vendor/runtime libraries.
