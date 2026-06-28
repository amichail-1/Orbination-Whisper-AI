package main

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

type RuntimeProfile struct {
	Name                   string
	Description            string
	BeamSize               int
	BestOf                 int
	TemperatureInc         float64
	NoContext              bool
	GPUWorkers             int
	CPUWorkers             int
	CPUAssist              int
	FlashAttn              bool
	PreferredModelPatterns []string
}

func RuntimeProfileByName(name string) (RuntimeProfile, error) {
	switch normalizeProfileName(name) {
	case "quality", "best", "default", "q4", "q4k":
		return RuntimeProfile{
			Name:           "quality",
			Description:    "best shippable small/fast/quality tradeoff; prefers baked Q4_K and beam search",
			BeamSize:       5,
			BestOf:         1,
			TemperatureInc: 0.2,
			NoContext:      true,
			GPUWorkers:     1,
			CPUWorkers:     0,
			CPUAssist:      1,
			FlashAttn:      true,
			PreferredModelPatterns: []string{
				"baked*q4_k", "baked*q4", "q4_k*baked", "q4*baked",
				"large-v3-turbo*q4_k", "large-v3-turbo*q4", "turbo*q4_k", "turbo*q4",
				"baked*q5_k", "large-v3-turbo*q5_k", "turbo*q5_k",
				"large-v3-turbo", "turbo",
				"baked*q3_k", "large-v3-turbo*q3_k", "turbo*q3_k",
			},
		}, nil
	case "balanced", "balance":
		return RuntimeProfile{
			Name:           "balanced",
			Description:    "faster than quality with small accuracy loss; prefers Q4_K and moderate beam",
			BeamSize:       3,
			BestOf:         1,
			TemperatureInc: 0.2,
			NoContext:      true,
			GPUWorkers:     1,
			CPUWorkers:     0,
			CPUAssist:      1,
			FlashAttn:      true,
			PreferredModelPatterns: []string{
				"baked*q4_k", "large-v3-turbo*q4_k", "turbo*q4_k",
				"baked*q3_k", "large-v3-turbo*q3_k", "turbo*q3_k",
				"baked*q5_k", "large-v3-turbo*q5_k", "turbo*q5_k",
			},
		}, nil
	case "speed", "fast":
		return RuntimeProfile{
			Name:           "speed",
			Description:    "lowest latency; prefers Q3_K and greedy decode",
			BeamSize:       1,
			BestOf:         1,
			TemperatureInc: 0.0,
			NoContext:      true,
			GPUWorkers:     1,
			CPUWorkers:     0,
			CPUAssist:      1,
			FlashAttn:      true,
			PreferredModelPatterns: []string{
				"baked*q3_k", "large-v3-turbo*q3_k", "turbo*q3_k", "q3_k",
				"baked*q4_k", "large-v3-turbo*q4_k", "turbo*q4_k", "q4_k",
			},
		}, nil
	case "tiny", "small", "smallest", "q3", "q3k":
		return RuntimeProfile{
			Name:           "tiny",
			Description:    "smallest package; prefers Q3_K; quality is lower than Q4_K-baked",
			BeamSize:       1,
			BestOf:         1,
			TemperatureInc: 0.0,
			NoContext:      true,
			GPUWorkers:     1,
			CPUWorkers:     0,
			CPUAssist:      1,
			FlashAttn:      true,
			PreferredModelPatterns: []string{
				"baked*q3_k", "large-v3-turbo*q3_k", "turbo*q3_k", "q3_k",
				"baked*q4_k", "large-v3-turbo*q4_k", "turbo*q4_k", "q4_k",
			},
		}, nil
	case "server", "throughput":
		return RuntimeProfile{
			Name:           "server",
			Description:    "throughput-oriented HTTP mode; GPU first and CPU assists during backlog",
			BeamSize:       5,
			BestOf:         1,
			TemperatureInc: 0.2,
			NoContext:      true,
			GPUWorkers:     1,
			CPUWorkers:     2,
			CPUAssist:      1,
			FlashAttn:      true,
			PreferredModelPatterns: []string{
				"baked*q4_k", "large-v3-turbo*q4_k", "turbo*q4_k", "q4_k",
				"baked*q3_k", "large-v3-turbo*q3_k", "turbo*q3_k", "q3_k",
			},
		}, nil
	case "list", "help":
		return RuntimeProfile{}, fmt.Errorf("available profiles: quality, balanced, speed, tiny, server")
	default:
		return RuntimeProfile{}, fmt.Errorf("unknown profile %q; use quality, balanced, speed, tiny, or server", name)
	}
}

func normalizeProfileName(name string) string {
	name = strings.TrimSpace(strings.ToLower(name))
	name = strings.ReplaceAll(name, "_", "-")
	return name
}

func AutoSelectModel(profile RuntimeProfile) (string, error) {
	if env := strings.TrimSpace(os.Getenv("WHISPERHYBRID_MODEL")); env != "" {
		return env, nil
	}

	candidates := discoverModelFiles()
	if len(candidates) == 0 {
		return "", fmt.Errorf("-model auto found no .bin/.gguf model; put baked-q4_k.bin or ggml-large-v3-turbo-q4_k.bin under ./model or ./app/model, or pass -model PATH")
	}

	for _, pat := range profile.PreferredModelPatterns {
		for _, c := range candidates {
			base := strings.ToLower(filepath.Base(c))
			if matchLoosePattern(base, strings.ToLower(pat)) {
				return c, nil
			}
		}
	}

	return candidates[0], nil
}

func discoverModelFiles() []string {
	seen := map[string]bool{}
	var dirs []string

	if wd, err := os.Getwd(); err == nil {
		dirs = append(dirs, filepath.Join(wd, "model"), filepath.Join(wd, "app", "model"))
	}
	if exe, err := os.Executable(); err == nil {
		dirs = append(dirs, filepath.Join(filepath.Dir(exe), "model"))
	}
	dirs = append(dirs, "model", filepath.Join("app", "model"))

	var out []string
	for _, dir := range dirs {
		entries, err := os.ReadDir(dir)
		if err != nil {
			continue
		}
		for _, e := range entries {
			if e.IsDir() {
				continue
			}
			name := strings.ToLower(e.Name())
			if !(strings.HasSuffix(name, ".bin") || strings.HasSuffix(name, ".gguf")) {
				continue
			}
			p := filepath.Clean(filepath.Join(dir, e.Name()))
			if seen[p] {
				continue
			}
			seen[p] = true
			out = append(out, p)
		}
	}
	sort.Strings(out)
	return out
}

func matchLoosePattern(name, pattern string) bool {
	parts := strings.Split(pattern, "*")
	pos := 0
	for _, part := range parts {
		if part == "" {
			continue
		}
		idx := strings.Index(name[pos:], part)
		if idx < 0 {
			return false
		}
		pos += idx + len(part)
	}
	return true
}
