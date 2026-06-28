# Q3_K-matched QAT (how the 0.148 model was made)

The key: the QAT fake-quant calls the EXACT ggml Q3_K (via ctypes) with a straight-through
estimator, so training == deployment. Plain fine-tune + Q3_K gives 0.197; matched-QAT gives 0.148.

1. Build the exact-Q3_K ctypes lib (links ggml):
   gcc -O2 -fPIC -fopenmp -shared quant_lib.c -I<whisper.cpp>/ggml/include \
       -L<whisper.cpp>/build/bin -lggml -lggml-base -o libq3k.so
2. Run QAT (PyTorch, one-time, offline) from the FP16 fine-tuned checkpoint:
   python q3k_qat.py        # wraps enc+dec linears, exact Q3_K STE + teacher KD
3. Export (no PyTorch at runtime): save FP16 -> ggml -> whisper-quantize q3_k.
   The exported standard Q3_K GGUF runs in the Go app at the same WER (~0.148).
