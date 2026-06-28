//go:build cuda

package main

/*
// CUDA runtime libs live outside the linker's default path; add common locations.
// Override with CGO_LDFLAGS="-L/your/cuda/lib64" if your install differs.
#cgo linux LDFLAGS: -L${SRCDIR}/lib -L/usr/local/cuda/lib64 -L/usr/local/cuda-12.8/lib64 -L/usr/lib/x86_64-linux-gnu -lwhisper -lggml -lggml-base -lggml-cpu -lggml-cuda -lm -lstdc++ -lcudart -lcublas -lcublasLt
#cgo windows LDFLAGS: -L${SRCDIR}/lib -lwhisper -lggml -lggml-base -lggml-cpu -lggml-cuda -lstdc++ -lcudart -lcublas -lcublasLt
*/
import "C"
