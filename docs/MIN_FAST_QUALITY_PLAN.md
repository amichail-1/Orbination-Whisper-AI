# Smallest + fastest + best quality plan

These three goals conflict, so the release must expose **profiles** rather than pretending one setting can maximize everything.

## The default answer

Use **large-v3-turbo Q4_K-baked** as the default model.

Why:

- It is much smaller than FP16.
- It keeps the Greek fine-tune/QAT gain better than plain Q3_K.
- It runs through stock `whisper.cpp` kernels, so the Go runtime stays portable.
- It is the best current shippable point from the project results: about **474 MB** and **0.189 Greek WER**.

Plain **Q3_K** remains available for the smallest package, but it is not the quality target: the project results show about **368 MB** and **0.285 Greek WER** for plain Q3_K.

## Runtime profiles now supported by the Go app

The app has a `-profile` flag:

```bash
./whisperhybrid -profile quality  -model auto -lang el clip.wav
./whisperhybrid -profile balanced -model auto -lang el clip.wav
./whisperhybrid -profile speed    -model auto -lang el clip.wav
./whisperhybrid -profile tiny     -model auto -lang el clip.wav
./whisperhybrid -profile server   -model auto -serve :8080 -lang el
```

| Profile | Preferred model | Decode | Use this when |
|---|---|---|---|
| `quality` | baked Q4_K | beam 5 | Best shippable quality per MB. This is the default. |
| `balanced` | baked Q4_K | beam 3 | Lower latency with small quality loss. |
| `speed` | Q3_K | greedy | Lowest latency. |
| `tiny` | Q3_K | greedy | Smallest package. |
| `server` | baked Q4_K | beam 5, GPU + CPU assist | HTTP throughput under bursts. |

`-model auto` searches `./model`, `./app/model`, and a `model` folder next to the executable. It prefers names like:

```text
baked-q4_k.bin
baked-q4_k.gguf
ggml-large-v3-turbo-q4_k.bin
ggml-large-v3-turbo-q3_k.bin
```

You can always override the profile:

```bash
./whisperhybrid -profile quality -model model/baked-q4_k.bin -beam 1 clip.wav
./whisperhybrid -profile tiny -model model/ggml-large-v3-turbo-q3_k.bin -beam 5 clip.wav
```

## Recommended artifacts to ship

Do not ship every model by default. Ship one primary artifact and one optional tiny artifact.

```text
recommended release:
  whisperhybrid(.exe)
  model/baked-q4_k.bin
  lib/...

optional tiny release:
  whisperhybrid(.exe)
  model/ggml-large-v3-turbo-q3_k.bin
  lib/...
```

## Commands

Quality default:

```bash
./whisperhybrid -profile quality -model auto -lang el clip.wav
```

Fastest/tiny:

```bash
./whisperhybrid -profile tiny -model auto -lang el clip.wav
```

Server:

```bash
./whisperhybrid -profile server -model auto -serve :8080 -lang el
```

Benchmark the profiles on a held-out manifest:

```bash
python3 tools/profile_sweep.py \
  --manifest fleurs-el-heldout.csv \
  --bin ./app/whisperhybrid \
  --lang el \
  --models-dir ./app/model \
  --out profile_sweep.csv
```

## Next frontier

To get close to **Q3_K size** while keeping **near Q4_K-baked quality**, the next model work should be **Q3_K-matched QAT**:

```text
training fake quantizer == whisper.cpp Q3_K deployment quantizer
```

That path can keep using stock whisper.cpp CPU/CUDA/Metal/Vulkan kernels. A custom INT3/outlier ggml type can preserve the current custom kernel more exactly, but it requires C/CUDA/Metal/Vulkan kernel work and produces a larger engineering surface.
