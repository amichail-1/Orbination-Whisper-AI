#!/usr/bin/env bash
set -euo pipefail

# Build whisper.cpp libraries for the current Unix-like host and copy headers/libs
# into app/inc and app/lib. Run from the repository root.
#
# Usage:
#   scripts/build_whispercpp_unix.sh cpu
#   scripts/build_whispercpp_unix.sh metal
#   scripts/build_whispercpp_unix.sh cuda
#   scripts/build_whispercpp_unix.sh vulkan
#   scripts/build_whispercpp_unix.sh rocm
#
# Set WHISPER_CPP_DIR=/path/to/whisper.cpp to reuse an existing checkout.

backend="${1:-cpu}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
whisper_dir="${WHISPER_CPP_DIR:-$root/external/whisper.cpp}"

if [[ ! -d "$whisper_dir/.git" ]]; then
  mkdir -p "$(dirname "$whisper_dir")"
  git clone https://github.com/ggml-org/whisper.cpp.git "$whisper_dir"
fi

build_dir="$whisper_dir/build-$backend"
cmake_args=(
  -DCMAKE_BUILD_TYPE=Release
  -DBUILD_SHARED_LIBS=OFF
)

case "$backend" in
  cpu) ;;
  metal) cmake_args+=(-DGGML_METAL=ON) ;;
  cuda) cmake_args+=(-DGGML_CUDA=ON) ;;
  vulkan) cmake_args+=(-DGGML_VULKAN=ON) ;;
  rocm) cmake_args+=(-DGGML_HIP=ON) ;;
  *) echo "unknown backend: $backend" >&2; exit 2 ;;
esac

cmake -S "$whisper_dir" -B "$build_dir" "${cmake_args[@]}"
cmake --build "$build_dir" --config Release -j

mkdir -p "$root/app/lib" "$root/app/inc/ggml"
cp "$whisper_dir/include/whisper.h" "$root/app/inc/"
cp "$whisper_dir/ggml/include/"*.h "$root/app/inc/ggml/"

# Different CMake versions/backends place archives in slightly different folders.
find "$build_dir" -type f \( -name 'libwhisper.*' -o -name 'libggml*.*' \) -exec cp -f {} "$root/app/lib/" \;

echo "Copied whisper.cpp headers and libraries to app/inc and app/lib"
