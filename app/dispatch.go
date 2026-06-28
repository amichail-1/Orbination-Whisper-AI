package main

import (
	"fmt"
	"os"
	"sync"
	"sync/atomic"
	"time"
)

type Job struct {
	Samples  []float32
	Lang     string
	Enqueued time.Time
	Result   chan JobResult
}

type JobResult struct {
	Text    string
	Err     error
	Backend string // "GPU" or "CPU"
	Millis  int64
}

type DispatcherConfig struct {
	ModelPath  string
	GPUWorkers int
	CPUWorkers int
	Threads    int
	CPUAssist  int
	GPUDevice  int
	StrictGPU  bool
	FlashAttn  bool
	Decode     DecodeOptions
}

// Dispatcher routes jobs to GPU and CPU engines. CPU workers stay out of the way
// while the queue is small, then assist under backlog or when a job has waited.
type Dispatcher struct {
	queue      chan *Job
	pending    int64
	cpuAssist  int64
	gpuWorkers int64
	gpuDone    int64
	cpuDone    int64
	wg         sync.WaitGroup
	closeOnce  sync.Once
	engines    []*Engine
}

func NewDispatcher(cfg DispatcherConfig) (*Dispatcher, error) {
	if cfg.Threads < 1 {
		cfg.Threads = 1
	}
	if cfg.GPUWorkers < 0 || cfg.CPUWorkers < 0 {
		return nil, fmt.Errorf("worker counts must be >= 0")
	}
	if cfg.GPUWorkers == 0 && cfg.CPUWorkers == 0 {
		cfg.CPUWorkers = 1
	}

	d := &Dispatcher{queue: make(chan *Job, 1024), cpuAssist: int64(cfg.CPUAssist)}

	for i := 0; i < cfg.GPUWorkers; i++ {
		e, err := NewEngine(EngineConfig{
			ModelPath: cfg.ModelPath,
			UseGPU:    true,
			GPUDevice: cfg.GPUDevice,
			Threads:   cfg.Threads,
			FlashAttn: cfg.FlashAttn,
			Decode:    cfg.Decode,
		})
		if err != nil {
			if cfg.StrictGPU {
				d.Close()
				return nil, fmt.Errorf("GPU worker %d init failed: %w", i, err)
			}
			fmt.Fprintf(os.Stderr, "[whisperhybrid] GPU worker %d init failed, continuing: %v\n", i, err)
			continue
		}
		d.engines = append(d.engines, e)
		atomic.AddInt64(&d.gpuWorkers, 1)
		d.wg.Add(1)
		go d.worker(e, true)
	}

	// If the requested GPU path did not come up and no CPU workers were requested,
	// fall back to one CPU worker instead of creating a dead server.
	if atomic.LoadInt64(&d.gpuWorkers) == 0 && cfg.CPUWorkers == 0 {
		cfg.CPUWorkers = 1
	}

	for i := 0; i < cfg.CPUWorkers; i++ {
		e, err := NewEngine(EngineConfig{
			ModelPath: cfg.ModelPath,
			UseGPU:    false,
			GPUDevice: cfg.GPUDevice,
			Threads:   cfg.Threads,
			FlashAttn: false,
			Decode:    cfg.Decode,
		})
		if err != nil {
			d.Close()
			return nil, fmt.Errorf("CPU worker %d init failed: %w", i, err)
		}
		d.engines = append(d.engines, e)
		d.wg.Add(1)
		go d.worker(e, false)
	}

	if len(d.engines) == 0 {
		return nil, fmt.Errorf("no whisper engines were started")
	}
	return d, nil
}

func (d *Dispatcher) worker(e *Engine, gpu bool) {
	defer d.wg.Done()
	for job := range d.queue {
		// CPU assist gate: only gate CPU when at least one GPU worker exists.
		// CPU-only mode must never requeue forever.
		if !gpu && atomic.LoadInt64(&d.gpuWorkers) > 0 && atomic.LoadInt64(&d.pending) <= d.cpuAssist && time.Since(job.Enqueued) < 250*time.Millisecond {
			select {
			case d.queue <- job:
				time.Sleep(2 * time.Millisecond)
				continue
			default:
				// Queue is full, so help immediately.
			}
		}

		atomic.AddInt64(&d.pending, -1)
		t0 := time.Now()
		txt, err := e.Transcribe(job.Samples, job.Lang)
		ms := time.Since(t0).Milliseconds()
		bk := "CPU"
		if gpu {
			bk = "GPU"
			atomic.AddInt64(&d.gpuDone, 1)
		} else {
			atomic.AddInt64(&d.cpuDone, 1)
		}
		job.Result <- JobResult{Text: txt, Err: err, Backend: bk, Millis: ms}
	}
}

func (d *Dispatcher) Submit(samples []float32, lang string) JobResult {
	job := &Job{Samples: samples, Lang: lang, Enqueued: time.Now(), Result: make(chan JobResult, 1)}
	atomic.AddInt64(&d.pending, 1)
	d.queue <- job
	return <-job.Result
}

func (d *Dispatcher) Stats() (gpu, cpu, pending int64) {
	return atomic.LoadInt64(&d.gpuDone), atomic.LoadInt64(&d.cpuDone), atomic.LoadInt64(&d.pending)
}

func (d *Dispatcher) Close() {
	if d == nil {
		return
	}
	d.closeOnce.Do(func() {
		close(d.queue)
		d.wg.Wait()
		for _, e := range d.engines {
			e.Close()
		}
	})
}
