#!/usr/bin/env bash
set -euo pipefail

# Build a practical release for the current project goal:
#   small + fast + best quality => baked Q4_K default, Q3_K optional tiny profile.
#
# Usage:
#   scripts/build_optimal_release_unix.sh BACKEND BASE_MODEL OUT_DIR
#
# Examples:
#   scripts/build_optimal_release_unix.sh metal models/baked-fp16.bin dist/macos-arm64-metal
#   scripts/build_optimal_release_unix.sh cuda  models/baked-fp16.bin dist/linux-amd64-cuda
#   scripts/build_optimal_release_unix.sh cpu   models/baked-fp16.bin dist/linux-amd64-cpu
#
# BACKEND: cpu | metal | cuda | vulkan | rocm
# BASE_MODEL may be an fp16/f32 ggml model. If q3_k/q4_k already exist, pass "-".

backend="${1:-cpu}"
base_model="${2:-}"
out_dir="${3:-dist/$(uname -s | tr '[:upper:]' '[:lower:]')-$(uname -m)-${backend}}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
models_dir="$root/app/model"

case "$backend" in
  cpu) tags="" ;;
  metal|cuda|vulkan|rocm) tags="$backend" ;;
  *) echo "unknown backend: $backend" >&2; exit 1 ;;
esac

mkdir -p "$models_dir" "$out_dir"

"$root/scripts/build_whispercpp_unix.sh" "$backend"

if [[ -n "$base_model" && "$base_model" != "-" ]]; then
  "$root/scripts/quantize_models.sh" "$base_model" "$models_dir"
fi

if [[ -n "$tags" ]]; then
  "$root/scripts/build_app_unix.sh" whisperhybrid -tags "$tags"
else
  "$root/scripts/build_app_unix.sh" whisperhybrid
fi

pick_model() {
  local pattern="$1"
  local hit
  hit="$(find "$models_dir" -maxdepth 1 -type f \( -name "*$pattern*.bin" -o -name "*$pattern*.gguf" \) | sort | head -n 1 || true)"
  [[ -n "$hit" ]] && printf '%s\n' "$hit"
}

quality_model="$(pick_model 'q4_k')"
tiny_model="$(pick_model 'q3_k')"

if [[ -z "$quality_model" ]]; then
  echo "No Q4_K model found under $models_dir. Put baked-q4_k.bin there or pass a base model to quantize." >&2
  exit 1
fi

rm -rf "$out_dir"
mkdir -p "$out_dir/model" "$out_dir/lib"
cp "$root/app/whisperhybrid" "$out_dir/"
cp "$quality_model" "$out_dir/model/"
if [[ -n "$tiny_model" ]]; then
  cp "$tiny_model" "$out_dir/model/"
fi
cp "$root/app/lib"/* "$out_dir/lib/" 2>/dev/null || true

cat > "$out_dir/run-quality.sh" <<'RUN'
#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
export LD_LIBRARY_PATH="$DIR/lib:${LD_LIBRARY_PATH:-}"
export DYLD_LIBRARY_PATH="$DIR/lib:${DYLD_LIBRARY_PATH:-}"
exec "$DIR/whisperhybrid" -profile quality -model auto "$@"
RUN
chmod +x "$out_dir/run-quality.sh"

cat > "$out_dir/run-tiny.sh" <<'RUN'
#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
export LD_LIBRARY_PATH="$DIR/lib:${LD_LIBRARY_PATH:-}"
export DYLD_LIBRARY_PATH="$DIR/lib:${DYLD_LIBRARY_PATH:-}"
exec "$DIR/whisperhybrid" -profile tiny -model auto "$@"
RUN
chmod +x "$out_dir/run-tiny.sh"

cat > "$out_dir/README.txt" <<TXT
WhisperHybrid release

Default quality command:
  ./run-quality.sh -lang el audio.wav

Tiny/fast command:
  ./run-tiny.sh -lang el audio.wav

Server:
  ./whisperhybrid -profile server -model auto -serve :8080 -lang el
TXT

echo "release written to $out_dir"
