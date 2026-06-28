#!/usr/bin/env bash
set -euo pipefail

# Build the Go application on Linux/macOS after app/lib contains whisper.cpp libs.
# Usage:
#   scripts/build_app_unix.sh whisperhybrid
#   scripts/build_app_unix.sh whisperhybrid -tags embedded_model

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out="${1:-whisperhybrid}"
shift || true

cd "$root/app"
CGO_ENABLED=1 go build "$@" -trimpath -ldflags="-s -w" -o "$out" .
echo "built app/$out"
