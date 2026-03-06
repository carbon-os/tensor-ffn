// ============================================================
// FILE: trainer/src/model/rmsnorm.cpp
// ============================================================
#include "model/rmsnorm.h"
#include "backend/cuda_ops.h"

namespace model {

void rmsnorm_forward(const __nv_bfloat16* x, const __nv_bfloat16* w,
                     __nv_bfloat16* out, float* rms_scratch,
                     int rows, int dim, float eps) {
    backend::rmsnorm_bf16(x, w, out, rms_scratch, rows, dim, eps);
}

void rmsnorm_backward(const float* d_out, const __nv_bfloat16* x,
                      const __nv_bfloat16* w, const float* rms_scratch,
                      float* d_x, int rows, int dim, float /*eps*/) {
    backend::rmsnorm_backward_fp32(d_out, x, w, rms_scratch, d_x, rows, dim);
}

} // namespace model