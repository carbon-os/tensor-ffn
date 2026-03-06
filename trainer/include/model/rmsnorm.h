// ============================================================
// FILE: trainer/include/model/rmsnorm.h
// ============================================================
#pragma once
#include <cuda_bf16.h>
#include <cublas_v2.h>

namespace model {

// Stateless helpers wrapping backend::rmsnorm_bf16 / backward.
// Weights stay as bf16 pointers owned by the caller (QwenLayer or ExpertFFN).

// Forward: normalises x, writes to out, saves rms scratch for backward.
void rmsnorm_forward(const __nv_bfloat16* x,
                     const __nv_bfloat16* w,
                     __nv_bfloat16*       out,
                     float*               rms_scratch,
                     int rows, int dim, float eps);

// Backward: given upstream gradient d_out and saved {x, w, rms}, compute d_x.
void rmsnorm_backward(const float*         d_out,
                      const __nv_bfloat16* x,
                      const __nv_bfloat16* w,
                      const float*         rms_scratch,
                      float*               d_x,
                      int rows, int dim, float eps);

} // namespace model