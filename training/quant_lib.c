#include "ggml.h"
#include <stdlib.h>
#include <omp.h>
// exact ggml Q3_K round-trip, parallelized over rows (n_per_row multiple of 256)
void q3k_rt(const float* in, float* out, long nrows, long n_per_row){
    size_t rs = ggml_row_size(GGML_TYPE_Q3_K, n_per_row);
    const struct ggml_type_traits* tr = ggml_get_type_traits(GGML_TYPE_Q3_K);
    #pragma omp parallel
    {
        void* q = malloc(rs);
        #pragma omp for
        for (long r=0;r<nrows;r++){
            ggml_quantize_chunk(GGML_TYPE_Q3_K, in+r*n_per_row, q, 0, 1, n_per_row, NULL);
            tr->to_float(q, out+r*n_per_row, n_per_row);
        }
        free(q);
    }
}
