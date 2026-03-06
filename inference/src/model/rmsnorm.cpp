#include "model/qwen.h"
#include "backend/cuda_ops.h"

namespace model {

// Thin wrapper — rms_buf may be nullptr if caller does not need the saved scale.
void rmsnorm_forward(const __nv_bfloat16* x, const __nv_bfloat16* w,
                     __nv_bfloat16* out, float* rms_scratch,
                     int rows, int dim, float eps) {
    backend::rmsnorm_bf16(x, w, out, rms_scratch, rows, dim, eps);
}

} // namespace model