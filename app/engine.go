package main

/*
#cgo CFLAGS: -I${SRCDIR}/inc -I${SRCDIR}/inc/ggml
#include <stdlib.h>
#include <string.h>
#include "whisper.h"

static struct whisper_context* init_model(const char* path, int use_gpu, int gpu_device, int flash_attn) {
    struct whisper_context_params cp = whisper_context_default_params();
    cp.use_gpu = use_gpu != 0;
    cp.gpu_device = gpu_device;
    cp.flash_attn = flash_attn != 0;
    return whisper_init_from_file_with_params(path, cp);
}

// transcribe returns a malloc'd C string; caller frees it. NULL means whisper_full failed.
static char* transcribe(struct whisper_context* ctx, float* samples, int n,
                        const char* lang, int nthreads, int beam_size, int best_of,
                        float temperature_inc, int no_context, int no_timestamps,
                        int single_segment, int translate, int print_timings) {
    enum whisper_sampling_strategy strategy = beam_size > 1 ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY;
    struct whisper_full_params p = whisper_full_default_params(strategy);

    if (beam_size > 1) {
        p.beam_search.beam_size = beam_size;
    } else {
        p.greedy.best_of = best_of > 0 ? best_of : 1;
    }

    p.language = lang;
    p.detect_language = lang == NULL || lang[0] == 0 || strcmp(lang, "auto") == 0;
    p.n_threads = nthreads > 0 ? nthreads : 1;
    p.no_timestamps = no_timestamps != 0;
    p.print_progress = false;
    p.print_realtime = false;
    p.print_special = false;
    p.print_timestamps = false;
    p.translate = translate != 0;
    p.single_segment = single_segment != 0;
    p.temperature_inc = temperature_inc;
    p.entropy_thold = 2.4f;
    p.no_context = no_context != 0;

    if (whisper_full(ctx, p, samples, n) != 0) return NULL;
    if (print_timings != 0) whisper_print_timings(ctx);

    int ns = whisper_full_n_segments(ctx);
    size_t cap = 256;
    char* out = (char*)malloc(cap);
    if (out == NULL) return NULL;
    out[0] = 0;
    size_t len = 0;

    for (int i = 0; i < ns; i++) {
        const char* t = whisper_full_get_segment_text(ctx, i);
        size_t tl = strlen(t);
        while (len + tl + 1 > cap) {
            cap *= 2;
            char* next = (char*)realloc(out, cap);
            if (next == NULL) {
                free(out);
                return NULL;
            }
            out = next;
        }
        memcpy(out + len, t, tl);
        len += tl;
        out[len] = 0;
    }
    return out;
}
*/
import "C"
import (
	"errors"
	"fmt"
	"unsafe"
)

type DecodeOptions struct {
	BeamSize       int
	BestOf         int
	TemperatureInc float32
	NoContext      bool
	NoTimestamps   bool
	SingleSegment  bool
	Translate      bool
	PrintTimings   bool
}

func DefaultDecodeOptions() DecodeOptions {
	return DecodeOptions{
		BeamSize:       5,
		BestOf:         1,
		TemperatureInc: 0.2,
		NoContext:      true,
		NoTimestamps:   true,
	}
}

type EngineConfig struct {
	ModelPath string
	UseGPU    bool
	GPUDevice int
	Threads   int
	FlashAttn bool
	Decode    DecodeOptions
}

// Engine wraps one whisper context. A context must not be used concurrently, so
// the dispatcher creates one context per worker.
type Engine struct {
	ctx     *C.struct_whisper_context
	gpu     bool
	threads int
	decode  DecodeOptions
}

func NewEngine(cfg EngineConfig) (*Engine, error) {
	if cfg.ModelPath == "" {
		return nil, errors.New("model path is empty")
	}
	if cfg.Threads < 1 {
		cfg.Threads = 1
	}
	cfg.Decode = normalizeDecodeOptions(cfg.Decode)

	cpath := C.CString(cfg.ModelPath)
	defer C.free(unsafe.Pointer(cpath))
	useGPU := C.int(0)
	if cfg.UseGPU {
		useGPU = 1
	}
	flash := C.int(0)
	if cfg.FlashAttn {
		flash = 1
	}
	ctx := C.init_model(cpath, useGPU, C.int(cfg.GPUDevice), flash)
	if ctx == nil {
		return nil, fmt.Errorf("failed to load model %q", cfg.ModelPath)
	}
	return &Engine{ctx: ctx, gpu: cfg.UseGPU, threads: cfg.Threads, decode: cfg.Decode}, nil
}

func normalizeDecodeOptions(o DecodeOptions) DecodeOptions {
	if o.BeamSize < 1 {
		o.BeamSize = 1
	}
	if o.BestOf < 1 {
		o.BestOf = 1
	}
	if o.TemperatureInc < 0 {
		o.TemperatureInc = 0
	}
	return o
}

func (e *Engine) Transcribe(samples []float32, lang string) (string, error) {
	if len(samples) == 0 {
		return "", errors.New("empty audio")
	}
	if lang == "" {
		lang = "auto"
	}
	clang := C.CString(lang)
	defer C.free(unsafe.Pointer(clang))

	noContext := boolToCInt(e.decode.NoContext)
	noTimestamps := boolToCInt(e.decode.NoTimestamps)
	singleSegment := boolToCInt(e.decode.SingleSegment)
	translate := boolToCInt(e.decode.Translate)
	printTimings := boolToCInt(e.decode.PrintTimings)

	res := C.transcribe(e.ctx, (*C.float)(unsafe.Pointer(&samples[0])), C.int(len(samples)),
		clang, C.int(e.threads), C.int(e.decode.BeamSize), C.int(e.decode.BestOf),
		C.float(e.decode.TemperatureInc), noContext, noTimestamps,
		singleSegment, translate, printTimings)
	if res == nil {
		return "", errors.New("transcription failed")
	}
	defer C.free(unsafe.Pointer(res))
	return C.GoString(res), nil
}

func boolToCInt(v bool) C.int {
	if v {
		return 1
	}
	return 0
}

func (e *Engine) Close() {
	if e != nil && e.ctx != nil {
		C.whisper_free(e.ctx)
		e.ctx = nil
	}
}

func (e *Engine) IsGPU() bool { return e.gpu }
