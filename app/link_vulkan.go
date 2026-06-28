//go:build vulkan

package main

/*
#cgo linux LDFLAGS: -L${SRCDIR}/lib -lwhisper -lggml -lggml-base -lggml-cpu -lggml-vulkan -lvulkan -lm -lstdc++
#cgo darwin LDFLAGS: -L${SRCDIR}/lib -lwhisper -lggml -lggml-base -lggml-cpu -lggml-vulkan -lvulkan -lc++ -framework Foundation -framework Accelerate
#cgo windows LDFLAGS: -L${SRCDIR}/lib -lwhisper -lggml -lggml-base -lggml-cpu -lggml-vulkan -lvulkan -lstdc++
*/
import "C"
