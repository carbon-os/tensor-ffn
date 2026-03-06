#include "model/expert.h"
#include "backend/memory.h"
#include "backend/cuda_ops.h"
#include <cuda_runtime.h>
#include <cstdio>

namespace model {

void expert_alloc(ExpertModel& e, int num_layers, int hidden_size,
                  int intermediate_size, int max_tokens) {
    e.num_layers        = num_layers;
    e.hidden_size       = hidden_size;
    e.intermediate_size = intermediate_size;
    e.max_tokens        = max_tokens;
    e.output_scale      = kExpertOutputScale;

    size_t gu = (size_t)intermediate_size * hidden_size;
    size_t d  = (size_t)hidden_size * intermediate_size;

    for (int l = 0; l < num_layers; ++l) {
        auto& el = e.layers[l];
        el.gate_proj = (__nv_bfloat16*)backend::device_alloc(gu * sizeof(__nv_bfloat16));
        el.up_proj   = (__nv_bfloat16*)backend::device_alloc(gu * sizeof(__nv_bfloat16));
        el.down_proj = (__nv_bfloat16*)backend::device_alloc(d  * sizeof(__nv_bfloat16));
    }

    size_t act = (size_t)max_tokens * intermediate_size;
    e.gate_buf = (__nv_bfloat16*)backend::device_alloc(act * sizeof(__nv_bfloat16));
    e.up_buf   = (__nv_bfloat16*)backend::device_alloc(act * sizeof(__nv_bfloat16));
    e.act_buf  = (__nv_bfloat16*)backend::device_alloc(act * sizeof(__nv_bfloat16));
}

void expert_free(ExpertModel& e) {
    for (int l = 0; l < e.num_layers; ++l) {
        backend::device_free(e.layers[l].gate_proj);
        backend::device_free(e.layers[l].up_proj);
        backend::device_free(e.layers[l].down_proj);
    }
    backend::device_free(e.gate_buf);
    backend::device_free(e.up_buf);
    backend::device_free(e.act_buf);
}

void expert_forward_layer(ExpertModel& e, cublasHandle_t cublas, int layer,
                           const __nv_bfloat16* x_norm_l,
                           float* delta_l,
                           int T) {
    auto& el = e.layers[layer];
    int H = e.hidden_size, I = e.intermediate_size;

    backend::gemm_bf16_out(cublas, T, I, H, x_norm_l, el.gate_proj, e.gate_buf, false, true);
    backend::gemm_bf16_out(cublas, T, I, H, x_norm_l, el.up_proj,   e.up_buf,   false, true);
    backend::silu_mul_bf16(e.gate_buf, e.up_buf, e.act_buf, T * I);
    backend::gemm_bf16(cublas, T, H, I,
                       e.output_scale,   // ← runtime scale, not compile-time constant
                       e.act_buf, el.down_proj,
                       0.f, delta_l,
                       false, true);
}

} // namespace model