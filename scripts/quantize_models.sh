#!/usr/bin/env bash
set -euo pipefail

# Quantize a Whisper ggml model into the common deployment variants.
# Usage:
#   WHISPER_CPP_DIR=external/whisper.cpp scripts/quantize_models.sh models/baked-fp16.bin models
#
# Optional:
#   WHISPER_QUANTIZE=/path/to/quantize scripts/quantize_models.sh ...

base="${1:?base fp16/f32 ggml model required}"
outdir="${2:?output directory required}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
whisper_dir="${WHISPER_CPP_DIR:-$root/external/whisper.cpp}"

find_quantize() {
  if [[ -n "${WHISPER_QUANTIZE:-}" && -x "${WHISPER_QUANTIZE:-}" ]]; then
    printf '%s\n' "$WHISPER_QUANTIZE"
    return 0
  fi
  local candidates=(
    "$whisper_dir/build/bin/quantize"
    "$whisper_dir/build-cpu/bin/quantize"
    "$whisper_dir/build-metal/bin/quantize"
    "$whisper_dir/build-cuda/bin/quantize"
    "$whisper_dir/build-vulkan/bin/quantize"
    "$whisper_dir/build-rocm/bin/quantize"
  )
  local c
  for c in "${candidates[@]}"; do
    if [[ -x "$c" ]]; then
      printf '%s\n' "$c"
      return 0
    fi
  done
  find "$whisper_dir" -path '*/bin/quantize' -type f -perm -111 2>/dev/null | sort | head -n 1
}

quant="$(find_quantize)"
if [[ -z "$quant" || ! -x "$quant" ]]; then
  echo "quantize binary not found under $whisper_dir" >&2
  echo "Run scripts/build_whispercpp_unix.sh cpu first, or set WHISPER_CPP_DIR / WHISPER_QUANTIZE." >&2
  exit 1
fi

mkdir -p "$outdir"
stem="$(basename "$base")"
stem="${stem%.*}"

for q in q3_k q4_k q5_k; do
  "$quant" "$base" "$outdir/${stem}-${q}.bin" "$q"
done

echo "wrote q3_k/q4_k/q5_k models to $outdir"
