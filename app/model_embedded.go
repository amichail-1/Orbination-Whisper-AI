//go:build embedded_model

package main

import (
	"crypto/sha256"
	"embed"
	"encoding/hex"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
)

// Put exactly one .bin/.gguf model under app/model/ and build with:
//
//	go build -tags embedded_model -o whisperhybrid .
//
// The model is extracted to the user cache before whisper.cpp opens it.
//
//go:embed model/*
var embeddedModelFS embed.FS

func resolveModel(path string) (string, func(), error) {
	if path != ":embedded" {
		return path, func() {}, nil
	}
	entries, err := fs.ReadDir(embeddedModelFS, "model")
	if err != nil {
		return "", func() {}, err
	}
	var modelName string
	for _, e := range entries {
		name := e.Name()
		lower := strings.ToLower(name)
		if !e.IsDir() && (strings.HasSuffix(lower, ".bin") || strings.HasSuffix(lower, ".gguf")) {
			if modelName != "" {
				return "", func() {}, fmt.Errorf("embedded_model expects exactly one model file under app/model")
			}
			modelName = name
		}
	}
	if modelName == "" {
		return "", func() {}, fmt.Errorf("no embedded .bin/.gguf model found under app/model")
	}
	data, err := embeddedModelFS.ReadFile("model/" + modelName)
	if err != nil {
		return "", func() {}, err
	}
	sum := sha256.Sum256(data)
	cache, err := os.UserCacheDir()
	if err != nil {
		cache = os.TempDir()
	}
	dir := filepath.Join(cache, "whisperhybrid")
	if err := os.MkdirAll(dir, 0755); err != nil {
		return "", func() {}, err
	}
	out := filepath.Join(dir, hex.EncodeToString(sum[:8])+"-"+modelName)
	if st, err := os.Stat(out); err == nil && st.Size() == int64(len(data)) {
		return out, func() {}, nil
	}
	if err := os.WriteFile(out, data, 0644); err != nil {
		return "", func() {}, err
	}
	return out, func() {}, nil
}
