#pragma once
#include <cuda_bf16.h>
#include <cublas_v2.h>

namespace model {

struct ExpertLayer {
    __nv_bfloat16* gate_proj = nullptr; // [I, H]
    __nv_bfloat16* up_proj   = nullptr;
    __nv_bfloat16* down_proj = nullptr; // [H, I]
};

struct ExpertModel {
    int num_layers        = 0;
    int hidden_size       = 0;
    int intermediate_size = 0;
    int max_tokens        = 0;

    // Runtime-adjustable — overrides the baked-in constant at inference.
    float output_scale = 0.1f;

    ExpertLayer layers[32];

    __nv_bfloat16* gate_buf = nullptr; // [max_tokens, I]
    __nv_bfloat16* up_buf   = nullptr;
    __nv_bfloat16* act_buf  = nullptr;
};

static constexpr float kExpertOutputScale = 0.1f; // default / training value

void expert_alloc(ExpertModel& e, int num_layers, int hidden_size,
                  int intermediate_size, int max_tokens);
void expert_free(ExpertModel& e);

// delta_l[T, H] fp32 = output_scale * SwiGLU(x_norm_l) @ down^T
void expert_forward_layer(ExpertModel& e, cublasHandle_t cublas, int layer,
                          const __nv_bfloat16* x_norm_l,
                          float* delta_l,
                          int T);

} // namespace model