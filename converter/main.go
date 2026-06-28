// qbake: pure-Go converter. Reads QAT model (safetensors), applies OUR custom
// quantization (group-16 INT3/INT4 + outliers) to encoder linears, and writes a
// legacy-ggml whisper model that runs in whisper.cpp / the Go app. NO PyTorch.
package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
	"strings"
)

// ---------- float helpers ----------
func bf16tof32(v uint16) float32 { return math.Float32frombits(uint32(v) << 16) }
func f16tof32(h uint16) float32 {
	s := uint32(h&0x8000) << 16
	e := uint32(h>>10) & 0x1f
	m := uint32(h & 0x3ff)
	if e == 0 {
		if m == 0 {
			return math.Float32frombits(s)
		}
		for m&0x400 == 0 {
			m <<= 1
			e--
		}
		e++
		m &= 0x3ff
	} else if e == 0x1f {
		return math.Float32frombits(s | 0x7f800000 | m<<13)
	}
	return math.Float32frombits(s | (e+112)<<23 | m<<13)
}
func f32tof16(f float32) uint16 {
	b := math.Float32bits(f)
	s := uint16((b >> 16) & 0x8000)
	e := int32((b>>23)&0xff) - 127 + 15
	m := b & 0x7fffff
	if e <= 0 {
		if e < -10 {
			return s
		}
		m |= 0x800000
		sh := uint32(14 - e)
		return s | uint16((m+(1<<(sh-1)))>>sh)
	} else if e >= 0x1f {
		return s | 0x7c00
	}
	return s | uint16(e)<<10 | uint16((m+0x1000)>>13)
}

// ---------- safetensors ----------
type stInfo struct {
	Dtype string  `json:"dtype"`
	Shape []int   `json:"shape"`
	Off   []int64 `json:"data_offsets"`
}
type Safet struct {
	f       *os.File
	infos   map[string]stInfo
	dataPos int64
}

func openSafet(path string) (*Safet, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	var n uint64
	binary.Read(f, binary.LittleEndian, &n)
	hb := make([]byte, n)
	io.ReadFull(f, hb)
	raw := map[string]json.RawMessage{}
	json.Unmarshal(hb, &raw)
	infos := map[string]stInfo{}
	for k, v := range raw {
		if k == "__metadata__" {
			continue
		}
		var si stInfo
		if json.Unmarshal(v, &si) == nil {
			infos[k] = si
		}
	}
	return &Safet{f: f, infos: infos, dataPos: int64(8 + n)}, nil
}

func (s *Safet) read(name string) ([]float32, bool) {
	si, ok := s.infos[name]
	if !ok {
		return nil, false
	}
	sz := si.Off[1] - si.Off[0]
	buf := make([]byte, sz)
	s.f.ReadAt(buf, s.dataPos+si.Off[0])
	var out []float32
	switch si.Dtype {
	case "BF16":
		out = make([]float32, sz/2)
		for i := range out {
			out[i] = bf16tof32(binary.LittleEndian.Uint16(buf[i*2:]))
		}
	case "F16":
		out = make([]float32, sz/2)
		for i := range out {
			out[i] = f16tof32(binary.LittleEndian.Uint16(buf[i*2:]))
		}
	case "F32":
		out = make([]float32, sz/4)
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(buf[i*4:]))
		}
	default:
		return nil, false
	}
	return out, true
}

// ---------- OUR quantization (group-wise INT-k + magnitude outliers) ----------
func quantize(w []float32, rows, cols, bits, gs int, outlierFrac float64) {
	qmax := float32(int(1)<<(bits-1) - 1) // 4-bit->7, 3-bit->3
	// outlier threshold (top frac by |w| kept as-is)
	var thr float32 = math.MaxFloat32
	if outlierFrac > 0 {
		mags := make([]float32, len(w))
		for i, x := range w {
			mags[i] = absf(x)
		}
		sort.Slice(mags, func(i, j int) bool { return mags[i] < mags[j] })
		idx := int(float64(len(mags)) * (1 - outlierFrac))
		if idx >= 0 && idx < len(mags) {
			thr = mags[idx]
		}
	}
	if cols%gs != 0 {
		gs = cols
	}
	for r := 0; r < rows; r++ {
		base := r * cols
		for g0 := 0; g0 < cols; g0 += gs {
			// per-group amax
			var amax float32
			for j := g0; j < g0+gs; j++ {
				a := absf(w[base+j])
				if a > amax {
					amax = a
				}
			}
			if amax < 1e-8 {
				amax = 1e-8
			}
			scale := amax / qmax
			for j := g0; j < g0+gs; j++ {
				if absf(w[base+j]) >= thr {
					continue
				} // outlier: keep FP
				q := float32(math.Round(float64(w[base+j] / scale)))
				if q > qmax {
					q = qmax
				} else if q < -qmax {
					q = -qmax
				}
				w[base+j] = q * scale
			}
		}
	}
}
func absf(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}

// ---------- HF name -> ggml name ----------
var convMap = map[string]string{
	"self_attn.k_proj": "attn.key", "self_attn.q_proj": "attn.query",
	"self_attn.v_proj": "attn.value", "self_attn.out_proj": "attn.out",
	"self_attn_layer_norm": "attn_ln",
	"encoder_attn.q_proj":  "cross_attn.query", "encoder_attn.v_proj": "cross_attn.value",
	"encoder_attn.out_proj": "cross_attn.out", "encoder_attn_layer_norm": "cross_attn_ln",
	"fc1": "mlp.0", "fc2": "mlp.2", "final_layer_norm": "mlp_ln",
	"encoder.embed_positions.weight": "encoder.positional_embedding",
	"decoder.embed_positions.weight": "decoder.positional_embedding",
	"decoder.embed_tokens.weight":    "decoder.token_embedding.weight",
}

func hf2ggml(name string) string {
	if name == "proj_out.weight" {
		return ""
	}
	nn := strings.Split(name, ".")[1:] // drop "model"
	if len(nn) >= 2 && nn[1] == "layers" {
		nn[1] = "blocks"
		mid := strings.Join(nn[3:len(nn)-1], ".")
		var mapped string
		if mid == "encoder_attn.k_proj" {
			if nn[0] == "encoder" {
				mapped = "attn.key"
			} else {
				mapped = "cross_attn.key"
			}
		} else {
			mapped = convMap[mid]
			if mapped == "" {
				return ""
			}
		}
		out := append(append(append([]string{}, nn[:3]...), mapped), nn[len(nn)-1])
		return strings.Join(out, ".")
	}
	j := strings.Join(nn, ".")
	if m, ok := convMap[j]; ok {
		return m
	}
	return j
}

// is this ggml tensor a linear weight we quantize? returns bits (0=no).
// FULL mode (env QBAKE_FULL=1): quantize encoder AND decoder linears at 3-bit
// (matches full-model 3-bit QAT). Else: encoder-only (q/k=4, rest=3).
func quantBits(ggml string) int {
	full := os.Getenv("QBAKE_FULL") == "1"
	isBlock := strings.HasPrefix(ggml, "encoder.blocks.")
	if full {
		isBlock = isBlock || strings.HasPrefix(ggml, "decoder.blocks.")
	}
	if !isBlock || !strings.HasSuffix(ggml, ".weight") {
		return 0
	}
	isLinear := strings.Contains(ggml, ".attn.") || strings.Contains(ggml, ".cross_attn.") ||
		strings.Contains(ggml, ".mlp.0") || strings.Contains(ggml, ".mlp.2")
	if !isLinear {
		return 0
	}
	if full {
		return 3
	} // full QAT used 3-bit everywhere
	if strings.Contains(ggml, ".attn.query") || strings.Contains(ggml, ".attn.key") {
		return 4
	}
	return 3
}

// ---------- legacy ggml parsing ----------
type tens struct {
	name   string
	ftype  int32
	dims   []int32 // actual shape (already un-reversed)
	off    int64   // data start
	nbytes int64
}

func parseGGML(path string) (int64, []tens, error) {
	f, err := os.Open(path)
	if err != nil {
		return 0, nil, err
	}
	defer f.Close()
	r := func(n int64) int64 { p, _ := f.Seek(0, io.SeekCurrent); _ = p; return n }
	_ = r
	var i32 int32
	rd := func() int32 { binary.Read(f, binary.LittleEndian, &i32); return i32 }
	if rd() != 0x67676d6c {
		return 0, nil, fmt.Errorf("bad magic")
	}
	for i := 0; i < 11; i++ {
		rd()
	} // 11 hparams after magic (vocab..use_f16)
	nmel := rd()
	nfft := rd()
	f.Seek(int64(nmel)*int64(nfft)*4, io.SeekCurrent) // mel filters
	ntok := rd()
	for i := int32(0); i < ntok; i++ {
		l := rd()
		f.Seek(int64(l), io.SeekCurrent)
	}
	var ts []tens
	for {
		var nd int32
		if err := binary.Read(f, binary.LittleEndian, &nd); err != nil {
			break
		}
		nl := rd()
		ft := rd()
		dims := make([]int32, nd)
		var ne int64 = 1
		for i := int32(0); i < nd; i++ {
			d := rd()
			dims[nd-1-i] = d
			ne *= int64(d)
		}
		nb := make([]byte, nl)
		io.ReadFull(f, nb)
		esz := int64(4)
		if ft == 1 {
			esz = 2
		}
		pos, _ := f.Seek(0, io.SeekCurrent)
		ts = append(ts, tens{string(nb), ft, dims, pos, ne * esz})
		f.Seek(ne*esz, io.SeekCurrent)
	}
	return 0, ts, nil
}

func main() {
	if len(os.Args) != 4 {
		fmt.Fprintf(os.Stderr, "usage: %s base-fp16-ggml.bin qat-model.safetensors output-ggml.bin\n", os.Args[0])
		os.Exit(2)
	}
	base := os.Args[1] // base legacy ggml (fp16) of the model
	stp := os.Args[2]  // QAT safetensors
	out := os.Args[3]  // output ggml
	// copy base -> out
	in, _ := os.ReadFile(base)
	os.WriteFile(out, in, 0644)
	_, ts, err := parseGGML(base)
	if err != nil {
		fmt.Println("parse error:", err)
		os.Exit(1)
	}
	st, err := openSafet(stp)
	if err != nil {
		fmt.Println("safet error:", err)
		os.Exit(1)
	}
	// ggml-name -> hf-name
	g2h := map[string]string{}
	for hf := range st.infos {
		if g := hf2ggml(hf); g != "" {
			g2h[g] = hf
		}
	}
	of, _ := os.OpenFile(out, os.O_RDWR, 0644)
	defer of.Close()
	patched, quantized, missing := 0, 0, 0
	for _, t := range ts {
		hf, ok := g2h[t.name]
		if !ok {
			missing++
			continue
		}
		vals, ok := st.read(hf)
		if !ok {
			missing++
			continue
		}
		if b := quantBits(t.name); b > 0 && len(t.dims) == 2 {
			rows, cols := int(t.dims[0]), int(t.dims[1])
			if rows*cols == len(vals) {
				quantize(vals, rows, cols, b, 16, 0.005)
				quantized++
			}
		}
		// encode to ftype and write at offset
		var buf []byte
		if t.ftype == 1 {
			buf = make([]byte, len(vals)*2)
			for i, v := range vals {
				binary.LittleEndian.PutUint16(buf[i*2:], f32tof16(v))
			}
		} else {
			buf = make([]byte, len(vals)*4)
			for i, v := range vals {
				binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
			}
		}
		if int64(len(buf)) == t.nbytes {
			of.WriteAt(buf, t.off)
			patched++
		} else {
			missing++
		}
	}
	fmt.Printf("[qbake] patched=%d quantized(enc linears)=%d skipped=%d -> %s\n", patched, quantized, missing, out)
}
