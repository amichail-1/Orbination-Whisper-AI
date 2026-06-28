#!/usr/bin/env bash
cd "$(dirname "$0")"
LD_LIBRARY_PATH=./lib ./whisperhybrid -profile server -model model/ggml-large-v3-turbo-q3_k.bin -serve "${1:-:8080}" -lang "${2:-el}"
