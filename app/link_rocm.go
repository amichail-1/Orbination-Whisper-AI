//go:build rocm

package main

/*
#cgo linux LDFLAGS: -L${SRCDIR}/lib -lwhisper -lggml -lggml-base -lggml-cpu -lggml-hip -lm -lstdc++ -lamdhip64 -lhipblas
*/
import "C"
