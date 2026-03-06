// ============================================================
// FILE: trainer/include/model/ffn_expert.h
// ============================================================
#pragma once
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cstdint>
#include <string>

namespace model {

// One expert FFN for a single transformer layer.
// Shape mirrors Qwen 2.5 SwiGLU FFN:
//   gate_proj [intermediate, hidden]
//   up_proj   [intermediate, hidden]
//   down_proj [hidden, intermediate]
// Output = down_proj(silu(gate_proj(x)) * up_proj(x))
// This output is ADDED to the frozen base FFN output (not a replacement).
struct ExpertLayer {
    // bf16 working weights (used in matmul)
    __nv_bfloat16* gate_proj = nullptr;
    __nv_bfloat16* up_proj   = nullptr;
    __nv_bfloat16* down_proj = nullptr;

    // fp32 master weights (optimizer updates these, then casts to bf16)
    float* gate_proj_fp32 = nullptr;
    float* up_proj_fp32   = nullptr;
    float* down_proj_fp32 = nullptr;

    // AdamW 1st moment
    float* gate_m1 = nullptr;
    float* up_m1   = nullptr;
    float* down_m1 = nullptr;

    // AdamW 2nd moment
    float* gate_m2 = nullptr;
    float* up_m2   = nullptr;
    float* down_m2 = nullptr;

    // Gradient accumulators (fp32)
    float* gate_grad = nullptr;
    float* up_grad   = nullptr;
    float* down_grad = nullptr;
};

// Collection of 24 per-layer experts.
struct ExpertFFN {
    static constexpr int MAX_LAYERS = 24;
    ExpertLayer layers[MAX_LAYERS];
    int  num_layers;
    int  hidden_size;
    int  intermediate_size;

    // Temporary activation buffers allocated once for the whole batch.
    // Reused every forward/backward call.
    // Layout: [num_layers, batch*seq, intermediate]
    __nv_bfloat16* gate_buf   = nullptr;  // pre-silu gate activations (bf16)
    __nv_bfloat16* up_buf     = nullptr;  // up activations             (bf16)
    __nv_bfloat16* gate_act   = nullptr;  // silu(gate)*up              (bf16)
    float*         d_gate_buf = nullptr;  // fp32 gradient on gate
    float*         d_up_buf   = nullptr;  // fp32 gradient on up
    float*         d_act_buf  = nullptr;  // fp32 gradient into act_in

    int max_tokens = 0;  // batch * seq_len allocated for
};

// Allocate all device memory. Call once before training.
// hidden and intermediate determined from config.
void expert_alloc(ExpertFFN& e, int num_layers,
                  int hidden_size, int intermediate_size,
                  int max_batch_tokens);

void expert_free(ExpertFFN& e);

// Xavier-uniform initialise weights (small scale so delta starts near zero).
void expert_init_weights(ExpertFFN& e, unsigned long seed = 42);

// Zero all gradient accumulators.
void expert_zero_grad(ExpertFFN& e);

// Forward: for each layer l, compute expert delta and write to out_delta[l].
// x_norm_per_layer: pointer to [num_layers, batch_tokens, hidden] bf16 tensor.
//   (Each slice is the FFN-normed input for that layer, precomputed by QwenModel.)
// out_delta: [num_layers, batch_tokens, hidden]  fp32  (added to frozen base FFN out)
void expert_forward(ExpertFFN&            e,
                    cublasHandle_t        cublas,
                    const __nv_bfloat16*  x_norm_per_layer, // [L, T, H]
                    float*                out_delta,         // [L, T, H]
                    int                   batch_tokens);

// Backward: given upstream d_delta [num_layers, T, H] (fp32),
// accumulate gradients into ExpertLayer.{gate,up,down}_grad.
// Also computes d_x_norm [L, T, H] (gradient w.r.t. the normed expert input).
void expert_backward(ExpertFFN&           e,
                     cublasHandle_t       cublas,
                     const __nv_bfloat16* x_norm_per_layer, // [L, T, H]  (saved fwd input)
                     const float*         d_delta,           // [L, T, H]  upstream grad
                     float*               d_x_norm,          // [L, T, H]  output grad
                     int                  batch_tokens);

} // namespace model