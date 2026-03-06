// ============================================================
// FILE: trainer/include/training/optimizer.h
// ============================================================
#pragma once
#include "model/ffn_expert.h"

namespace training {

struct AdamWConfig {
    float lr          = 3e-4f;
    float beta1       = 0.9f;
    float beta2       = 0.999f;
    float eps         = 1e-8f;
    float weight_decay = 0.1f;
    float grad_clip   = 1.0f;
};

struct AdamWState {
    AdamWConfig cfg;
    int         step = 0;
    // Cosine decay schedule
    int   total_steps     = 0;
    int   warmup_steps    = 100;
    float lr_min          = 0.0f;
};

void adamw_init(AdamWState& s, const AdamWConfig& cfg,
                int total_steps, int warmup_steps = 100);

// Compute current learning rate (cosine schedule with linear warmup).
float adamw_lr(const AdamWState& s);

// Clip global gradient norm across all expert layers.
// Returns the global grad norm before clipping.
float adamw_clip_grads(model::ExpertFFN& e, float max_norm, cudaStream_t s = 0);

// Apply one AdamW step to all expert layers.
// After updating fp32 master weights, casts to bf16.
void adamw_step(AdamWState& s, model::ExpertFFN& e, cudaStream_t stream = 0);

} // namespace training