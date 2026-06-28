package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"runtime"
	"time"
)

func main() {
	defaultThreads := runtime.NumCPU() / 2
	if defaultThreads < 1 {
		defaultThreads = 1
	}

	profileName := flag.String("profile", "quality", "runtime profile: quality|balanced|speed|tiny|server")
	model := flag.String("model", "auto", "path to ggml/quantized Whisper model, auto, or :embedded when built with -tags embedded_model")
	nGPU := flag.Int("gpu", 1, "number of GPU workers (0 to disable)")
	nCPU := flag.Int("cpu", 0, "number of CPU workers (0 to disable)")
	threads := flag.Int("threads", defaultThreads, "threads per worker")
	gpuDevice := flag.Int("gpu-device", 0, "GPU device index for CUDA-like backends")
	strictGPU := flag.Bool("strict-gpu", false, "fail startup if a requested GPU worker cannot initialize")
	flash := flag.Bool("flash", true, "request flash attention when the linked whisper.cpp backend supports it")
	lang := flag.String("lang", "auto", "language (auto|el|en|es|fr|...) default for CLI and server")
	serve := flag.String("serve", "", "HTTP listen addr, e.g. :8080 (empty = CLI mode)")
	assist := flag.Int("assist", 1, "CPU assists only when pending backlog > this; use 0 for fastest latency under mixed CPU+GPU")
	beam := flag.Int("beam", 5, "beam size: 1 = greedy/faster, 5 = better quality")
	bestOf := flag.Int("best-of", 1, "greedy best_of when -beam=1")
	tempInc := flag.Float64("temperature-inc", 0.2, "temperature fallback increment")
	noContext := flag.Bool("no-context", true, "disable cross-window context for independent clips")
	timestamps := flag.Bool("timestamps", false, "include timestamps internally; text output is still plain text")
	singleSegment := flag.Bool("single-segment", false, "force a single output segment")
	translate := flag.Bool("translate", false, "translate to English instead of transcribing")
	printTimings := flag.Bool("timings", false, "print whisper.cpp timing information")
	jsonCLI := flag.Bool("json", false, "emit one JSON object per input file in CLI mode")
	flag.Parse()
	setFlags := map[string]bool{}
	flag.Visit(func(f *flag.Flag) { setFlags[f.Name] = true })

	profile, err := RuntimeProfileByName(*profileName)
	if err != nil {
		fmt.Fprintln(os.Stderr, "profile error:", err)
		os.Exit(1)
	}
	if !setFlags["beam"] {
		*beam = profile.BeamSize
	}
	if !setFlags["best-of"] {
		*bestOf = profile.BestOf
	}
	if !setFlags["temperature-inc"] {
		*tempInc = profile.TemperatureInc
	}
	if !setFlags["no-context"] {
		*noContext = profile.NoContext
	}
	if !setFlags["gpu"] {
		*nGPU = profile.GPUWorkers
	}
	if !setFlags["cpu"] {
		*nCPU = profile.CPUWorkers
	}
	if !setFlags["assist"] {
		*assist = profile.CPUAssist
	}
	if !setFlags["flash"] {
		*flash = profile.FlashAttn
	}
	if *serve == "" && flag.NArg() == 0 {
		fmt.Println("usage: whisperhybrid [-profile quality|balanced|speed|tiny|server] [-model auto|MODEL] [-lang el] audio.wav [more.wav]")
		flag.PrintDefaults()
		return
	}
	if *model == "auto" {
		selected, err := AutoSelectModel(profile)
		if err != nil {
			fmt.Fprintln(os.Stderr, "model error:", err)
			os.Exit(1)
		}
		*model = selected
	}

	modelPath, cleanup, err := resolveModel(*model)
	if err != nil {
		fmt.Fprintln(os.Stderr, "model error:", err)
		os.Exit(1)
	}
	defer cleanup()

	decode := DefaultDecodeOptions()
	decode.BeamSize = *beam
	decode.BestOf = *bestOf
	decode.TemperatureInc = float32(*tempInc)
	decode.NoContext = *noContext
	decode.NoTimestamps = !*timestamps
	decode.SingleSegment = *singleSegment
	decode.Translate = *translate
	decode.PrintTimings = *printTimings

	fmt.Fprintf(os.Stderr, "[whisperhybrid] profile=%s model=%s gpu=%d cpu=%d threads=%d assist=%d beam=%d lang=%s\n",
		profile.Name, modelPath, *nGPU, *nCPU, *threads, *assist, *beam, *lang)

	d, err := NewDispatcher(DispatcherConfig{
		ModelPath:  modelPath,
		GPUWorkers: *nGPU,
		CPUWorkers: *nCPU,
		Threads:    *threads,
		CPUAssist:  *assist,
		GPUDevice:  *gpuDevice,
		StrictGPU:  *strictGPU,
		FlashAttn:  *flash,
		Decode:     decode,
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, "init error:", err)
		os.Exit(1)
	}
	defer d.Close()
	fmt.Fprintln(os.Stderr, "[whisperhybrid] ready")

	if *serve != "" {
		runServer(d, *serve, *lang)
		return
	}

	enc := json.NewEncoder(os.Stdout)
	for _, f := range flag.Args() {
		samples, err := LoadWAV(f)
		if err != nil {
			if *jsonCLI {
				_ = enc.Encode(map[string]any{"file": f, "text": "", "backend": "", "infer_ms": 0, "error": err.Error()})
			} else {
				fmt.Fprintln(os.Stderr, f, "load error:", err)
			}
			continue
		}
		r := d.Submit(samples, *lang)
		if *jsonCLI {
			_ = enc.Encode(map[string]any{"file": f, "text": r.Text, "backend": r.Backend, "infer_ms": r.Millis, "error": errStr(r.Err)})
			continue
		}
		if r.Err != nil {
			fmt.Fprintln(os.Stderr, f, "error:", r.Err)
			continue
		}
		fmt.Printf("\n[%s | %s | %dms] %s\n", f, r.Backend, r.Millis, r.Text)
	}
	g, c, _ := d.Stats()
	fmt.Fprintf(os.Stderr, "\n[stats] done GPU=%d CPU=%d\n", g, c)
}

func runServer(d *Dispatcher, addr, defLang string) {
	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, "ok")
	})
	mux.HandleFunc("/stats", func(w http.ResponseWriter, r *http.Request) {
		g, c, p := d.Stats()
		_ = json.NewEncoder(w).Encode(map[string]int64{"gpu_done": g, "cpu_done": c, "pending": p})
	})
	mux.HandleFunc("/inference", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "POST required", http.StatusMethodNotAllowed)
			return
		}
		// Keep the all-in-one server safe by rejecting huge uploads early.
		r.Body = http.MaxBytesReader(w, r.Body, 256<<20)
		file, _, err := r.FormFile("file")
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		defer file.Close()

		tmp, err := os.CreateTemp("", "wh-*.wav")
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		defer os.Remove(tmp.Name())
		if _, err := io.Copy(tmp, file); err != nil {
			_ = tmp.Close()
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if err := tmp.Close(); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		samples, err := LoadWAV(tmp.Name())
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		lang := r.FormValue("language")
		if lang == "" {
			lang = defLang
		}
		t0 := time.Now()
		res := d.Submit(samples, lang)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"text":     res.Text,
			"backend":  res.Backend,
			"infer_ms": res.Millis,
			"total_ms": time.Since(t0).Milliseconds(),
			"error":    errStr(res.Err),
		})
	})
	fmt.Fprintf(os.Stderr, "[whisperhybrid] serving on %s  (POST /inference file=@audio.wav)\n", addr)
	if err := http.ListenAndServe(addr, mux); err != nil {
		fmt.Fprintln(os.Stderr, "server error:", err)
	}
}

func errStr(e error) string {
	if e == nil {
		return ""
	}
	return e.Error()
}
