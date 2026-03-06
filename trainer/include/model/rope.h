// ============================================================
// FILE: trainer/include/model/rope.h
// ============================================================
#pragma once
#include <cuda_bf16.h>
#include <cstdint>

namespace model {

struct RopeTable {
    float* cos_d;   // device [max_seq, head_dim/2]
    float* sin_d;   // device [max_seq, head_dim/2]
    int    max_seq;
    int    head_dim;
    float  theta;
};

void rope_init(RopeTable& t, int max_seq, int head_dim, float theta);
void rope_free(RopeTable& t);

// Apply RoPE in-place to Q or K (bf16 layout).
//   x layout: [batch, seq, num_heads, head_dim]
//   positions: [seq]  (usually 0,1,...,seq-1)
void rope_apply_bf16(const RopeTable& t,
                     __nv_bfloat16*   x,
                     int              batch,
                     int              seq,
                     int              num_heads,
                     int              head_dim,
                     cudaStream_t     s = 0);

// fp32 variant used during backward recompute
void rope_apply_fp32(const RopeTable& t,
                     float*           x,
                     int              batch,
                     int              seq,
                     int              num_heads,
                     int              head_dim,
                     cudaStream_t     s = 0);

} // namespace model