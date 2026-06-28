//go:build metal

package main

/*
#cgo darwin LDFLAGS: -L${SRCDIR}/lib -lwhisper -lggml -lggml-base -lggml-cpu -lggml-metal -lc++ -framework Foundation -framework Accelerate -framework Metal -framework MetalKit -framework CoreGraphics -framework CoreVideo
*/
import "C"
