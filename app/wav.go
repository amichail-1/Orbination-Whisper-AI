package main

import (
	"encoding/binary"
	"errors"
	"fmt"
	"os"
)

// LoadWAV reads PCM16 RIFF WAV and returns mono float32 at 16 kHz.
// For MP3/M4A/FLAC, convert first, for example:
//
//	ffmpeg -i input.mp3 -ar 16000 -ac 1 -c:a pcm_s16le output.wav
func LoadWAV(path string) ([]float32, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(b) < 44 || string(b[0:4]) != "RIFF" || string(b[8:12]) != "WAVE" {
		return nil, errors.New("not a RIFF/WAVE file")
	}
	var channels, bits, audioFormat int
	var sr uint32
	var data []byte
	pos := 12
	for pos+8 <= len(b) {
		id := string(b[pos : pos+4])
		sz := int(binary.LittleEndian.Uint32(b[pos+4 : pos+8]))
		body := pos + 8
		if sz < 0 || body+sz > len(b)+1 {
			return nil, errors.New("corrupt WAV chunk")
		}
		switch id {
		case "fmt ":
			if body+16 <= len(b) {
				audioFormat = int(binary.LittleEndian.Uint16(b[body : body+2]))
				channels = int(binary.LittleEndian.Uint16(b[body+2 : body+4]))
				sr = binary.LittleEndian.Uint32(b[body+4 : body+8])
				bits = int(binary.LittleEndian.Uint16(b[body+14 : body+16]))
			}
		case "data":
			end := body + sz
			if end > len(b) {
				end = len(b)
			}
			data = b[body:end]
		}
		pos = body + sz
		if sz%2 == 1 {
			pos++
		}
	}
	if data == nil || channels == 0 {
		return nil, errors.New("missing WAV fmt/data chunk")
	}
	if audioFormat != 1 || bits != 16 {
		return nil, fmt.Errorf("unsupported WAV format: need PCM16, got format=%d bits=%d", audioFormat, bits)
	}
	if sr == 0 {
		return nil, errors.New("invalid WAV sample rate")
	}

	mono := make([]float32, 0, len(data)/(2*channels)+1)
	for i := 0; i+2*channels <= len(data); i += 2 * channels {
		var acc float32
		for c := 0; c < channels; c++ {
			s := int16(binary.LittleEndian.Uint16(data[i+2*c : i+2*c+2]))
			acc += float32(s) / 32768.0
		}
		mono = append(mono, acc/float32(channels))
	}
	if sr != 16000 {
		mono = resample(mono, int(sr), 16000)
	}
	return mono, nil
}

func resample(in []float32, from, to int) []float32 {
	if from == to || len(in) == 0 {
		return in
	}
	ratio := float64(to) / float64(from)
	out := make([]float32, int(float64(len(in))*ratio))
	for i := range out {
		src := float64(i) / ratio
		j := int(src)
		if j+1 < len(in) {
			f := float32(src - float64(j))
			out[i] = in[j]*(1-f) + in[j+1]*f
		} else if j < len(in) {
			out[i] = in[j]
		}
	}
	return out
}
