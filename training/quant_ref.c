#include "ggml.h"
#include <stdio.h>
#include <stdlib.h>
// stdin: n float32 ; stdout: n float32 (ggml Q3_K quantized->dequantized). argv[1]=n (mult of 256)
int main(int argc, char**argv){
    long n = atol(argv[1]);
    long npr = 256, nrows = n/npr;
    float* src = malloc(n*sizeof(float));
    if (fread(src,sizeof(float),n,stdin)!=(size_t)n){fprintf(stderr,"read err\n");return 1;}
    size_t qsz = ggml_row_size(GGML_TYPE_Q3_K, npr)*nrows;
    void* q = malloc(qsz);
    ggml_quantize_chunk(GGML_TYPE_Q3_K, src, q, 0, nrows, npr, NULL);
    float* out = malloc(n*sizeof(float));
    const struct ggml_type_traits* tr = ggml_get_type_traits(GGML_TYPE_Q3_K);
    tr->to_float(q, out, n);
    fwrite(out,sizeof(float),n,stdout);
    return 0;
}
