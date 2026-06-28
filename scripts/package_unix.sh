#!/usr/bin/env bash
set -euo pipefail

# Create a portable folder bundle: binary + whisper.cpp libs + optional model.
# Usage:
#   scripts/package_unix.sh dist/linux-amd64 ./app/whisperhybrid ./models/ggml-large-v3-turbo-q3_k.bin

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dist="${1:?dist directory required}"
bin="${2:?binary path required}"
model="${3:-}"

mkdir -p "$dist/lib" "$dist/model"
cp "$bin" "$dist/"
cp -f "$root/app/lib/"* "$dist/lib/" 2>/dev/null || true
if [[ -n "$model" ]]; then
  cp "$model" "$dist/model/"
fi
cat > "$dist/run.sh" <<'RUN'
#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH="$DIR/lib:${LD_LIBRARY_PATH:-}"
export DYLD_LIBRARY_PATH="$DIR/lib:${DYLD_LIBRARY_PATH:-}"
exec "$DIR/whisperhybrid" "$@"
RUN
chmod +x "$dist/run.sh"
echo "packaged $dist"
