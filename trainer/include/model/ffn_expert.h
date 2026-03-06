#pragma once
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cstdint>
#include <string>

namespace model {

static constexpr float kExpertOutputScale = 0.01f;

struct ExpertLayer {
    __nv_bfloat16* gate_proj = nullptr;
    __nv_bfloat16* up_proj   = nullptr;
    __nv_bfloat16* down_proj = nullptr;
    float* gate_proj_fp32 = nullptr;
    float* up_proj_fp32   = nullptr;
    float* down_proj_fp32 = nullptr;
    float* gate_m1 = nullptr; float* up_m1 = nullptr; float* down_m1 = nullptr;
    float* gate_m2 = nullptr; float* up_m2 = nullptr; float* down_m2 = nullptr;
    float* gate_grad = nullptr; float* up_grad = nullptr; float* down_grad = nullptr;
};

struct ExpertFFN {
    static constexpr int MAX_LAYERS = 24;
    ExpertLayer layers[MAX_LAYERS];
    int num_layers = 0, hidden_size = 0, intermediate_size = 0;
    __nv_bfloat16* gate_buf = nullptr;
    __nv_bfloat16* up_buf   = nullptr;
    __nv_bfloat16* gate_act = nullptr;
    float* d_gate_buf = nullptr;
    float* d_up_buf   = nullptr;
    float* d_act_buf  = nullptr;
    int max_tokens = 0;
};

void expert_alloc(ExpertFFN& e, int num_layers, int hidden_size,
                  int intermediate_size, int max_batch_tokens);
void expert_free(ExpertFFN& e);
void expert_init_weights(ExpertFFN& e, unsigned long seed = 42);
void expert_zero_grad(ExpertFFN& e);

void expert_forward(ExpertFFN& e, cublasHandle_t cublas,
                    const __nv_bfloat16* x_norm_per_layer,
                    float* out_delta, int batch_tokens);

// Declared AFTER ExpertFFN is complete
void expert_backward_layer(ExpertFFN& e, cublasHandle_t cublas, int layer,
                            const __nv_bfloat16* x_norm_l,
                            const float* d_delta_l,
                            float* d_x_norm_l,
                            int batch_tokens);

void expert_backward(ExpertFFN& e, cublasHandle_t cublas,
                     const __nv_bfloat16* x_norm_per_layer,
                     const float* d_delta, float* d_x_norm, int batch_tokens);

} // namespace model
