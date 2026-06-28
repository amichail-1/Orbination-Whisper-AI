#!/usr/bin/env bash
cd "$(dirname "$0")"
LD_LIBRARY_PATH=./lib ./whisperhybrid -profile quality -model model/ggml-large-v3-turbo-q3_k.bin -lang "${2:-el}" "${1:?usage: run-tiny.sh clip.wav [lang]}"
