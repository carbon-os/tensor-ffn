// ============================================================
// FILE: trainer/include/model/qwen.h
// ============================================================
#pragma once
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <string>
#include <cstdint>

namespace model {

// ---- config (read from config.json) ----
struct QwenConfig {
    int   hidden_size        = 896;
    int   intermediate_size  = 4864;
    int   num_layers         = 24;
    int   num_heads          = 14;
    int   num_kv_heads       = 2;
    int   head_dim           = 64;   // hidden_size / num_heads
    int   vocab_size         = 151936;
    int   max_seq_len        = 131072;
    float rms_norm_eps       = 1e-6f;
    float rope_theta         = 1e6f;
    int   eos_token_id       = 151645;
    int   bos_token_id       = 151643;
    bool  tie_embeddings     = true;
};

QwenConfig qwen_config_from_json(const std::string& model_dir);

// ---- Per-layer frozen weights ----
struct QwenLayer {
    // Attention projections  (all bf16, device)
    __nv_bfloat16* q_proj   = nullptr;  // [num_heads*head_dim, hidden]
    __nv_bfloat16* k_proj   = nullptr;  // [num_kv_heads*head_dim, hidden]
    __nv_bfloat16* v_proj   = nullptr;  // [num_kv_heads*head_dim, hidden]
    __nv_bfloat16* o_proj   = nullptr;  // [hidden, num_heads*head_dim]
    __nv_bfloat16* q_bias   = nullptr;  // [num_heads*head_dim]
    __nv_bfloat16* k_bias   = nullptr;  // [num_kv_heads*head_dim]
    __nv_bfloat16* v_bias   = nullptr;  // [num_kv_heads*head_dim]

    // RMSNorm weights (bf16)
    __nv_bfloat16* attn_norm = nullptr; // [hidden]
    __nv_bfloat16* ffn_norm  = nullptr; // [hidden]

    // Frozen base FFN weights (bf16)
    __nv_bfloat16* gate_proj = nullptr; // [intermediate, hidden]
    __nv_bfloat16* up_proj   = nullptr; // [intermediate, hidden]
    __nv_bfloat16* down_proj = nullptr; // [hidden, intermediate]
};

// ---- Full frozen model ----
struct QwenModel {
    QwenConfig cfg;
    static constexpr int MAX_LAYERS = 24;

    __nv_bfloat16* token_embed  = nullptr; // [vocab, hidden]
    QwenLayer      layers[MAX_LAYERS];
    __nv_bfloat16* final_norm   = nullptr; // [hidden]
    __nv_bfloat16* lm_head      = nullptr; // [vocab, hidden]  (may alias token_embed)
    bool           lm_head_tied = true;

    // RoPE tables (precomputed fp32)
    float* rope_cos = nullptr;  // [max_seq, head_dim/2]
    float* rope_sin = nullptr;  // [max_seq, head_dim/2]

    // Scratch buffers for attention (allocated to max_batch_tokens)
    float* attn_scratch = nullptr; // [batch*heads*seq*seq]  fp32
    float* qkv_scratch  = nullptr; // [batch*seq*(nq+2*nkv)*head_dim] fp32
    float* ffn_scratch  = nullptr; // [batch*seq*intermediate]  fp32
    int    max_tokens   = 0;
    int    max_seq      = 0;
};

// Load safetensors weights from model_dir into device memory.
void qwen_load(QwenModel& m, const std::string& model_dir);

// Free all device memory.
void qwen_free(QwenModel& m);

// Allocate scratch buffers sized for (max_batch, max_seq).
void qwen_alloc_scratch(QwenModel& m, int max_batch, int max_seq);

// ---- Forward pass ----
// Input:
//   tokens:       [batch, seq]  int32 on device
//   expert_delta: optional [num_layers, batch*seq, hidden] fp32 on device
//                 (added to each layer's FFN output; pass nullptr to skip)
// Output:
//   logits:       [batch*seq, vocab]  fp32 on device
// Saved for backward:
//   h_states:     [num_layers+1, batch*seq, hidden]  bf16 on device
//                  h_states[l] = residual stream entering layer l
//                  h_states[num_layers] = stream entering final_norm
//   ffn_norm_out: [num_layers, batch*seq, hidden]  bf16 on device
//                  output of ffn RMSNorm at each layer (expert's input)
//   ffn_rms:      [num_layers, batch*seq]  fp32  (for RMSNorm backward)
//   attn_rms:     [num_layers, batch*seq]  fp32
//   final_rms:    [batch*seq]  fp32
void qwen_forward(QwenModel&            m,
                  cublasHandle_t        cublas,
                  const int32_t*        tokens,        // [B, S]  device
                  int                   batch,
                  int                   seq,
                  const float*          expert_delta,  // [L, B*S, H] or nullptr
                  float*                logits,        // [B*S, V]  device  output
                  __nv_bfloat16*        h_states,      // [L+1, B*S, H]  saved
                  __nv_bfloat16*        ffn_norm_out,  // [L, B*S, H]  saved
                  float*                ffn_rms,       // [L, B*S]  saved
                  float*                attn_rms,      // [L, B*S]  saved
                  float*                final_rms);    // [B*S]  saved

// ---- Backward pass through frozen model ----
// Given d_logits [B*S, V] and saved activations, computes:
//   d_ffn_norm_out [L, B*S, H]: gradient w.r.t. the ffn-normed expert input
//                                (this is passed to expert_backward)
// The frozen weight gradients are NOT computed (they stay frozen).
void qwen_backward(QwenModel&            m,
                   cublasHandle_t        cublas,
                   const float*          d_logits,      // [B*S, V]
                   int                   batch,
                   int                   seq,
                   const __nv_bfloat16*  h_states,      // [L+1, B*S, H]  saved fwd
                   const __nv_bfloat16*  ffn_norm_out,  // [L, B*S, H]  saved fwd
                   const float*          ffn_rms,        // [L, B*S]
                   const float*          attn_rms,       // [L, B*S]
                   const float*          final_rms,      // [B*S]
                   const float*          d_expert_norm,  // [L, B*S, H] expert's d_x_norm
                   float*                d_h_scratch);   // [B*S, H]  temp buffer

} // namespace model