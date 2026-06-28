//go:build !cuda && !vulkan && !metal && !rocm

package main

/*
#cgo linux LDFLAGS: -L${SRCDIR}/lib -lwhisper -lggml -lggml-base -lggml-cpu -lm -lstdc++
#cgo darwin LDFLAGS: -L${SRCDIR}/lib -lwhisper -lggml -lggml-base -lggml-cpu -lc++ -framework Foundation -framework Accelerate
#cgo windows LDFLAGS: -L${SRCDIR}/lib -lwhisper -lggml -lggml-base -lggml-cpu -lstdc++
*/
import "C"
