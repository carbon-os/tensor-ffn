// ============================================================
// FILE: trainer/src/model/ffn_expert.cpp
// ============================================================
#include "model/ffn_expert.h"
#include "backend/memory.h"
#include "backend/cuda_ops.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <random>

namespace model {

static size_t expert_layer_param_bytes(const ExpertFFN& e, bool bf16) {
    size_t per = (size_t)e.hidden_size * e.intermediate_size;
    return 3 * per * (bf16 ? sizeof(__nv_bfloat16) : sizeof(float));
}

void expert_alloc(ExpertFFN& e, int num_layers,
                  int hidden_size, int intermediate_size,
                  int max_batch_tokens) {
    e.num_layers       = num_layers;
    e.hidden_size      = hidden_size;
    e.intermediate_size= intermediate_size;
    e.max_tokens       = max_batch_tokens;

    size_t gate_up = (size_t)intermediate_size * hidden_size;
    size_t down    = (size_t)hidden_size * intermediate_size;

    for (int l = 0; l < num_layers; ++l) {
        auto& el = e.layers[l];
        // bf16 working weights
        el.gate_proj = (__nv_bfloat16*)backend::device_alloc(gate_up * sizeof(__nv_bfloat16));
        el.up_proj   = (__nv_bfloat16*)backend::device_alloc(gate_up * sizeof(__nv_bfloat16));
        el.down_proj = (__nv_bfloat16*)backend::device_alloc(down    * sizeof(__nv_bfloat16));
        // fp32 master
        el.gate_proj_fp32 = (float*)backend::device_alloc(gate_up * sizeof(float));
        el.up_proj_fp32   = (float*)backend::device_alloc(gate_up * sizeof(float));
        el.down_proj_fp32 = (float*)backend::device_alloc(down    * sizeof(float));
        // moments
        el.gate_m1 = (float*)backend::device_alloc(gate_up * sizeof(float));
        el.gate_m2 = (float*)backend::device_alloc(gate_up * sizeof(float));
        el.up_m1   = (float*)backend::device_alloc(gate_up * sizeof(float));
        el.up_m2   = (float*)backend::device_alloc(gate_up * sizeof(float));
        el.down_m1 = (float*)backend::device_alloc(down    * sizeof(float));
        el.down_m2 = (float*)backend::device_alloc(down    * sizeof(float));
        // grads
        el.gate_grad = (float*)backend::device_alloc(gate_up * sizeof(float));
        el.up_grad   = (float*)backend::device_alloc(gate_up * sizeof(float));
        el.down_grad = (float*)backend::device_alloc(down    * sizeof(float));
        // zero moments
        backend::device_memset(el.gate_m1, 0, gate_up * sizeof(float));
        backend::device_memset(el.gate_m2, 0, gate_up * sizeof(float));
        backend::device_memset(el.up_m1,   0, gate_up * sizeof(float));
        backend::device_memset(el.up_m2,   0, gate_up * sizeof(float));
        backend::device_memset(el.down_m1, 0, down    * sizeof(float));
        backend::device_memset(el.down_m2, 0, down    * sizeof(float));
    }

    // Shared activation buffers (for one batch at a time)
    size_t act_bytes = (size_t)max_batch_tokens * intermediate_size;
    e.gate_buf   = (__nv_bfloat16*)backend::device_alloc(act_bytes * sizeof(__nv_bfloat16));
    e.up_buf     = (__nv_bfloat16*)backend::device_alloc(act_bytes * sizeof(__nv_bfloat16));
    e.gate_act   = (__nv_bfloat16*)backend::device_alloc(act_bytes * sizeof(__nv_bfloat16));
    e.d_gate_buf = (float*)        backend::device_alloc(act_bytes * sizeof(float));
    e.d_up_buf   = (float*)        backend::device_alloc(act_bytes * sizeof(float));
    e.d_act_buf  = (float*)        backend::device_alloc(act_bytes * sizeof(float));
}

void expert_free(ExpertFFN& e) {
    for (int l = 0; l < e.num_layers; ++l) {
        auto& el = e.layers[l];
        backend::device_free(el.gate_proj); backend::device_free(el.up_proj);
        backend::device_free(el.down_proj);
        backend::device_free(el.gate_proj_fp32); backend::device_free(el.up_proj_fp32);
        backend::device_free(el.down_proj_fp32);
        backend::device_free(el.gate_m1); backend::device_free(el.gate_m2);
        backend::device_free(el.up_m1);   backend::device_free(el.up_m2);
        backend::device_free(el.down_m1); backend::device_free(el.down_m2);
        backend::device_free(el.gate_grad); backend::device_free(el.up_grad);
        backend::device_free(el.down_grad);
    }
    backend::device_free(e.gate_buf); backend::device_free(e.up_buf);
    backend::device_free(e.gate_act);
    backend::device_free(e.d_gate_buf); backend::device_free(e.d_up_buf);
    backend::device_free(e.d_act_buf);
}

// Xavier uniform init
__global__ static void k_xavier_init(float* w, int n, float scale, unsigned seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // Simple LCG per thread
    unsigned s = seed ^ (unsigned)(i * 2654435761u);
    s = s * 1664525u + 1013904223u;
    float v = ((float)(s >> 16) / 65535.0f) * 2.0f - 1.0f;
    w[i] = v * scale;
}

void expert_init_weights(ExpertFFN& e, unsigned long seed) {
    int H = e.hidden_size, I = e.intermediate_size;

    // gate/up: Xavier uniform — these project input into intermediate space
    float scale_gate = sqrtf(6.0f / (float)(H + I));

    // down_proj: zero init — expert starts as a no-op delta
    // The branch output is expert_delta = gate_act @ down_proj.T
    // Zero down_proj → zero delta → base model runs unmodified at step 0
    // The optimizer will grow down_proj from zero as needed

    for (int l = 0; l < e.num_layers; ++l) {
        auto& el = e.layers[l];
        size_t gu = (size_t)I * H;
        size_t d  = (size_t)H * I;
        unsigned ls = (unsigned)(seed + l * 3);

        // gate_proj and up_proj: Xavier uniform (unchanged)
        k_xavier_init<<<(gu+255)/256, 256>>>(el.gate_proj_fp32, (int)gu, scale_gate, ls);
        k_xavier_init<<<(gu+255)/256, 256>>>(el.up_proj_fp32,   (int)gu, scale_gate, ls+1);

        // down_proj: zero
        backend::device_memset(el.down_proj_fp32, 0, d * sizeof(float));

        // Zero the fp32 master moments too (already done in expert_alloc,
        // but be explicit since init order matters)
        backend::device_memset(el.gate_m1, 0, gu * sizeof(float));
        backend::device_memset(el.gate_m2, 0, gu * sizeof(float));
        backend::device_memset(el.up_m1,   0, gu * sizeof(float));
        backend::device_memset(el.up_m2,   0, gu * sizeof(float));
        backend::device_memset(el.down_m1, 0, d  * sizeof(float));
        backend::device_memset(el.down_m2, 0, d  * sizeof(float));

        // Cast to bf16
        backend::cast_fp32_to_bf16(el.gate_proj_fp32, el.gate_proj, (int)gu);
        backend::cast_fp32_to_bf16(el.up_proj_fp32,   el.up_proj,   (int)gu);
        backend::cast_fp32_to_bf16(el.down_proj_fp32, el.down_proj, (int)d);
    }
    cudaDeviceSynchronize();
}


void expert_zero_grad(ExpertFFN& e) {
    for (int l = 0; l < e.num_layers; ++l) {
        auto& el = e.layers[l];
        size_t gu = (size_t)e.intermediate_size * e.hidden_size * sizeof(float);
        size_t d  = (size_t)e.hidden_size * e.intermediate_size * sizeof(float);
        backend::device_memset(el.gate_grad, 0, gu);
        backend::device_memset(el.up_grad,   0, gu);
        backend::device_memset(el.down_grad, 0, d);
    }
}

// Forward: for each layer compute expert delta
// x_norm_per_layer: [num_layers, batch_tokens, hidden]  bf16
// out_delta:        [num_layers, batch_tokens, hidden]   fp32
void expert_forward(ExpertFFN& e, cublasHandle_t cublas,
                    const __nv_bfloat16* x_norm_per_layer,
                    float* out_delta,
                    int batch_tokens) {
    int H = e.hidden_size, I = e.intermediate_size;
    size_t layer_x   = (size_t)batch_tokens * H;
    size_t layer_out = (size_t)batch_tokens * H;

    for (int l = 0; l < e.num_layers; ++l) {
        auto& el = e.layers[l];
        const __nv_bfloat16* x_l    = x_norm_per_layer + l * layer_x;
        float*               delta_l = out_delta        + l * layer_out;

        backend::gemm_bf16_out(cublas, batch_tokens, I, H,
                               x_l, el.gate_proj, e.gate_buf, false, true);
        backend::gemm_bf16_out(cublas, batch_tokens, I, H,
                               x_l, el.up_proj, e.up_buf, false, true);
        backend::silu_mul_bf16(e.gate_buf, e.up_buf, e.gate_act, batch_tokens * I);
        backend::gemm_bf16(cublas, batch_tokens, H, I,
                           1.0f, e.gate_act, el.down_proj,
                           0.0f, delta_l, false, true);

        backend::scale_fp32(delta_l, kExpertOutputScale, batch_tokens * H);
    }
}

void expert_backward_layer(ExpertFFN& e, cublasHandle_t cublas,
                            int layer,
                            const __nv_bfloat16* x_norm_l,
                            const float* d_delta_l,
                            float* d_x_norm_l,
                            int batch_tokens) {
    auto& el = e.layers[layer];
    int H = e.hidden_size, I = e.intermediate_size;

    // Backward through the output scale
    float* dd_scaled = (float*)backend::device_alloc(
        (size_t)batch_tokens * H * sizeof(float));
    backend::device_copy(dd_scaled, d_delta_l,
                         (size_t)batch_tokens * H * sizeof(float));
    backend::scale_fp32(dd_scaled, kExpertOutputScale, batch_tokens * H);

    // Recompute forward activations
    backend::gemm_bf16_out(cublas, batch_tokens, I, H,
                           x_norm_l, el.gate_proj, e.gate_buf, false, true);
    backend::gemm_bf16_out(cublas, batch_tokens, I, H,
                           x_norm_l, el.up_proj, e.up_buf, false, true);
    backend::silu_mul_bf16(e.gate_buf, e.up_buf, e.gate_act, batch_tokens * I);

    backend::cast_bf16_to_fp32(e.gate_buf, e.d_gate_buf, batch_tokens * I);
    backend::cast_bf16_to_fp32(e.up_buf,   e.d_up_buf,   batch_tokens * I);

    float* x_fp32 = (float*)backend::device_alloc(
        (size_t)batch_tokens * H * sizeof(float));
    backend::cast_bf16_to_fp32(x_norm_l, x_fp32, batch_tokens * H);

    // d_down_grad += gate_act.T @ dd_scaled
    backend::cast_bf16_to_fp32(e.gate_act, e.d_act_buf, batch_tokens * I);
    backend::gemm_fp32(cublas, H, I, batch_tokens,
                       1.0f, dd_scaled, e.d_act_buf,
                       1.0f, el.down_grad,
                       true, false);

    // d_gate_act = dd_scaled @ down_proj_fp32
    backend::gemm_fp32(cublas, batch_tokens, I, H,
                       1.0f, dd_scaled, el.down_proj_fp32,
                       0.0f, e.d_act_buf,
                       false, false);

    // SiLU-mul backward
    backend::silu_mul_backward_fp32(e.d_act_buf,
                                    e.d_gate_buf,
                                    e.d_up_buf,
                                    e.d_gate_buf,
                                    e.d_up_buf,
                                    batch_tokens * I);

    // Weight gradients
    backend::gemm_fp32(cublas, I, H, batch_tokens,
                       1.0f, e.d_gate_buf, x_fp32,
                       1.0f, el.gate_grad,
                       true, false);
    backend::gemm_fp32(cublas, I, H, batch_tokens,
                       1.0f, e.d_up_buf, x_fp32,
                       1.0f, el.up_grad,
                       true, false);

    // d_x_norm
    backend::gemm_fp32(cublas, batch_tokens, H, I,
                       1.0f, e.d_gate_buf, el.gate_proj_fp32,
                       0.0f, d_x_norm_l,
                       false, false);
    backend::gemm_fp32(cublas, batch_tokens, H, I,
                       1.0f, e.d_up_buf, el.up_proj_fp32,
                       1.0f, d_x_norm_l,
                       false, false);

    backend::device_free(dd_scaled);
    backend::device_free(x_fp32);
}


void expert_backward(ExpertFFN& e, cublasHandle_t cublas,
                     const __nv_bfloat16* x_norm_per_layer,
                     const float* d_delta,
                     float* d_x_norm,
                     int batch_tokens) {
    int H = e.hidden_size;
    size_t layer_stride = (size_t)batch_tokens * H;
    for (int l = 0; l < e.num_layers; ++l) {
        expert_backward_layer(e, cublas, l,
                              x_norm_per_layer + l * layer_stride,
                              d_delta          + l * layer_stride,
                              d_x_norm         + l * layer_stride,
                              batch_tokens);
    }
}

} // namespace model