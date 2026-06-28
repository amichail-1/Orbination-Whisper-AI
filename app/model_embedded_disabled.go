//go:build !embedded_model

package main

import "fmt"

func resolveModel(path string) (string, func(), error) {
	if path == ":embedded" {
		return "", func() {}, fmt.Errorf("this binary was not built with -tags embedded_model")
	}
	return path, func() {}, nil
}
