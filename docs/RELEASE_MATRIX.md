# Release matrix

Build one artifact per OS/architecture/backend. The Go code is portable, but the CGo-linked whisper.cpp libraries are platform-specific.

| Artifact | OS/arch | Backend | Go tags | Model recommendation |
|---|---|---|---|---|
| `whisperhybrid-macos-arm64-metal` | macOS ARM64 | Metal | `metal` | Q4_K-baked |
| `whisperhybrid-macos-amd64-cpu` | macOS Intel | CPU | none | Q4_K-baked or Q5_K |
| `whisperhybrid-linux-amd64-cuda` | Linux x86_64 | CUDA | `cuda` | Q4_K-baked, Q3_K small |
| `whisperhybrid-linux-amd64-vulkan` | Linux x86_64 | Vulkan | `vulkan` | Q4_K-baked |
| `whisperhybrid-linux-amd64-cpu` | Linux x86_64 | CPU AVX/AVX2 | none | Q4_K-baked |
| `whisperhybrid-linux-arm64-cpu` | Linux ARM64 | CPU NEON | none | Q3_K or Q4_K-baked |
| `whisperhybrid-windows-amd64-cuda.exe` | Windows x86_64 | CUDA | `cuda` | Q4_K-baked |
| `whisperhybrid-windows-amd64-vulkan.exe` | Windows x86_64 | Vulkan | `vulkan` | Q4_K-baked |
| `whisperhybrid-windows-amd64-cpu.exe` | Windows x86_64 | CPU | none | Q4_K-baked |

## Package layout

```text
whisperhybrid(.exe)
lib/
model/
  baked-q4_k.bin      # default quality profile
  *q3_k*.bin          # optional tiny profile
README_RUN.txt
run-quality.sh / run-tiny.sh or run.ps1
```

For support/debugging, include:

```text
BUILD_INFO.txt   OS, arch, backend, whisper.cpp commit, Go version
MODEL_INFO.txt   source checkpoint, quant type, WER table, eval manifest hash
```
