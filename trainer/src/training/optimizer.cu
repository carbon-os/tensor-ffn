// ============================================================
// FILE: trainer/src/training/optimizer.cpp
// ============================================================
#include "training/optimizer.h"
#include "backend/cuda_ops.h"
#include "backend/memory.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

namespace training {

void adamw_init(AdamWState& s, const AdamWConfig& cfg,
                int total_steps, int warmup_steps) {
    s.cfg         = cfg;
    s.step        = 0;
    s.total_steps = total_steps;
    s.warmup_steps = warmup_steps;
    s.lr_min      = cfg.lr * 0.1f;  // min LR for cosine schedule
}

float adamw_lr(const AdamWState& s) {
    int t = s.step;
    float base = s.cfg.lr;
    if (t < s.warmup_steps) {
        return base * (float)(t + 1) / (float)s.warmup_steps;
    }
    float progress = (float)(t - s.warmup_steps) /
                     (float)(s.total_steps - s.warmup_steps);
    float cosine   = 0.5f * (1.0f + cosf(3.14159265f * progress));
    return s.lr_min + (base - s.lr_min) * cosine;
}

// Compute global L2 grad norm across all expert layers
__global__ static void k_l2_sum(const float* g, float* out, int n) {
    float s = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += g[i] * g[i];
    __shared__ float ss[256]; ss[threadIdx.x] = s; __syncthreads();
    for (int d = blockDim.x>>1; d > 0; d >>= 1) {
        if (threadIdx.x < d) ss[threadIdx.x] += ss[threadIdx.x+d]; __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(out, ss[0]);
}

float adamw_clip_grads(model::ExpertFFN& e, float max_norm, cudaStream_t s) {
    float* norm_d; cudaMalloc(&norm_d, sizeof(float));
    cudaMemset(norm_d, 0, sizeof(float));

    for (int l = 0; l < e.num_layers; ++l) {
        auto& el = e.layers[l];
        int gu = e.intermediate_size * e.hidden_size;
        int d  = e.hidden_size * e.intermediate_size;
        k_l2_sum<<<1, 256, 0, s>>>(el.gate_grad, norm_d, gu);
        k_l2_sum<<<1, 256, 0, s>>>(el.up_grad,   norm_d, gu);
        k_l2_sum<<<1, 256, 0, s>>>(el.down_grad, norm_d, d);
    }

    float norm_h;
    cudaMemcpyAsync(&norm_h, norm_d, sizeof(float), cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);
    cudaFree(norm_d);
    norm_h = sqrtf(norm_h);

    if (norm_h > max_norm) {
        float scale = max_norm / (norm_h + 1e-6f);
        for (int l = 0; l < e.num_layers; ++l) {
            auto& el = e.layers[l];
            int gu = e.intermediate_size * e.hidden_size;
            int d  = e.hidden_size * e.intermediate_size;
            backend::scale_fp32(el.gate_grad, scale, gu, s);
            backend::scale_fp32(el.up_grad,   scale, gu, s);
            backend::scale_fp32(el.down_grad, scale, d,  s);
        }
    }
    return norm_h;
}

// AdamW update kernel for one weight tensor
__global__ static void k_adamw(float* w, float* m1, float* m2, const float* grad,
                                float lr, float beta1, float beta2,
                                float eps, float wd, float bc1, float bc2,
                                int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g  = grad[i];
    float m  = beta1 * m1[i] + (1.0f - beta1) * g;
    float v  = beta2 * m2[i] + (1.0f - beta2) * g * g;
    m1[i] = m; m2[i] = v;
    float mh = m / bc1;
    float vh = v / bc2;
    // weight decay applied to master weight
    w[i] = w[i] * (1.0f - lr * wd) - lr * mh / (sqrtf(vh) + eps);
}

void adamw_step(AdamWState& s, model::ExpertFFN& e, cudaStream_t stream) {
    s.step++;
    float lr   = adamw_lr(s);
    float beta1 = s.cfg.beta1, beta2 = s.cfg.beta2;
    float eps  = s.cfg.eps, wd = s.cfg.weight_decay;
    float bc1  = 1.0f - powf(beta1, (float)s.step);
    float bc2  = 1.0f - powf(beta2, (float)s.step);

    for (int l = 0; l < e.num_layers; ++l) {
        auto& el = e.layers[l];
        int gu = e.intermediate_size * e.hidden_size;
        int d  = e.hidden_size * e.intermediate_size;

        k_adamw<<<(gu+255)/256, 256, 0, stream>>>(
            el.gate_proj_fp32, el.gate_m1, el.gate_m2, el.gate_grad,
            lr, beta1, beta2, eps, wd, bc1, bc2, gu);
        k_adamw<<<(gu+255)/256, 256, 0, stream>>>(
            el.up_proj_fp32,   el.up_m1,   el.up_m2,   el.up_grad,
            lr, beta1, beta2, eps, wd, bc1, bc2, gu);
        k_adamw<<<(d +255)/256, 256, 0, stream>>>(
            el.down_proj_fp32, el.down_m1, el.down_m2, el.down_grad,
            lr, beta1, beta2, eps, wd, bc1, bc2, d);

        // Cast updated fp32 master weights → bf16 working weights
        backend::cast_fp32_to_bf16(el.gate_proj_fp32, el.gate_proj, gu, stream);
        backend::cast_fp32_to_bf16(el.up_proj_fp32,   el.up_proj,   gu, stream);
        backend::cast_fp32_to_bf16(el.down_proj_fp32, el.down_proj, d,  stream);
    }
}

} // namespace training