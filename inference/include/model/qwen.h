#pragma once
#include <string>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include "model/expert.h"

namespace model {

struct QwenConfig {
    int   hidden_size       = 896;
    int   intermediate_size = 4864;
    int   num_layers        = 24;
    int   num_heads         = 14;
    int   num_kv_heads      = 2;
    int   head_dim          = 64;
    int   vocab_size        = 151936;
    int   max_seq_len       = 131072;
    float rms_norm_eps      = 1e-6f;
    float rope_theta        = 1e6f;
    int   eos_token_id      = 151645;
    int   bos_token_id      = 151643;
    bool  tie_embeddings    = true;
};

struct QwenLayer {
    __nv_bfloat16* q_proj    = nullptr;
    __nv_bfloat16* k_proj    = nullptr;
    __nv_bfloat16* v_proj    = nullptr;
    __nv_bfloat16* o_proj    = nullptr;
    __nv_bfloat16* q_bias    = nullptr;
    __nv_bfloat16* k_bias    = nullptr;
    __nv_bfloat16* v_bias    = nullptr;
    __nv_bfloat16* attn_norm = nullptr;
    __nv_bfloat16* ffn_norm  = nullptr;
    __nv_bfloat16* gate_proj = nullptr;
    __nv_bfloat16* up_proj   = nullptr;
    __nv_bfloat16* down_proj = nullptr;
};

// KV cache — stores all past K and V in bf16.
// Layout: [num_layers, max_len, nkv_heads, head_dim]
struct KVCache {
    __nv_bfloat16* k      = nullptr; // [L, max_len, nkv, hd]
    __nv_bfloat16* v      = nullptr;
    int num_layers  = 0;
    int max_len     = 0;
    int nkv_heads   = 0;
    int head_dim    = 0;
    int len         = 0; // tokens currently stored

    __nv_bfloat16* k_layer(int l) {
        return k + (size_t)l * max_len * nkv_heads * head_dim;
    }
    __nv_bfloat16* v_layer(int l) {
        return v + (size_t)l * max_len * nkv_heads * head_dim;
    }
};

struct QwenModel {
    QwenConfig  cfg;
    QwenLayer   layers[32];
    __nv_bfloat16* token_embed = nullptr;
    __nv_bfloat16* final_norm  = nullptr;
    __nv_bfloat16* lm_head     = nullptr;
    bool lm_head_tied           = true;
    float* rope_cos = nullptr;
    float* rope_sin = nullptr;

    KVCache kv;

    // ---- pre-allocated scratch ----
    __nv_bfloat16* h_work    = nullptr; // [max_T, H]
    __nv_bfloat16* h_norm    = nullptr; // [max_T, H]
    __nv_bfloat16* q_bf16    = nullptr; // [max_T, nh, hd]
    __nv_bfloat16* k_bf16    = nullptr; // [max_T, nkv, hd]
    __nv_bfloat16* v_bf16    = nullptr; // [max_T, nkv, hd]
    float*         q_fp32    = nullptr; // [max_T, nh, hd]
    float*         k_fp32    = nullptr; // [max_T, nkv, hd]
    float*         k_fp32_ex = nullptr; // [max_total, nh, hd] expanded GQA
    float*         v_fp32_ex = nullptr; // [max_total, nh, hd]
    float*         q_perm    = nullptr; // [nh, max_T, hd]
    float*         k_perm    = nullptr; // [nh, max_total, hd]
    float*         v_perm    = nullptr; // [nh, max_total, hd]
    float*         scores    = nullptr; // [nh, max_T, max_total]
    float*         ctx_perm  = nullptr; // [nh, max_T, hd]
    float*         ctx_fp32  = nullptr; // [max_T, H]
    __nv_bfloat16* ctx_bf16  = nullptr; // [max_T, H]
    __nv_bfloat16* gate_buf  = nullptr; // [max_T, I]
    __nv_bfloat16* up_buf    = nullptr; // [max_T, I]
    __nv_bfloat16* act_buf   = nullptr; // [max_T, I]
    float*         ffn_fp32  = nullptr; // [max_T, H]
    float*         rms_buf   = nullptr; // [max_T]

    // expert delta buffer — [max_T, H] fp32 (one layer at a time)
    float*         expert_delta = nullptr;

    int max_T     = 0; // max tokens per forward call (= max_seq_len)
    int max_total = 0; // max_T (same, reused for KV total)
};

QwenConfig qwen_config_from_json(const std::string& model_dir);

// Load weights + allocate KV cache and scratch.
// max_seq_len is the maximum sequence (prompt + generation) length.
void qwen_load(QwenModel& m, const std::string& model_dir, int max_seq_len = 2048);
void qwen_free(QwenModel& m);

// Forward pass.  Appends T new tokens to the KV cache and returns their logits.
// expert may be nullptr — then base model runs alone.
void qwen_forward(QwenModel& m, cublasHandle_t cublas,
                  const int32_t* d_tokens, int T,
                  const ExpertModel* expert,
                  float* d_logits); // [T, vocab_size]

// Reset KV cache (call between independent prompts).
void qwen_reset(QwenModel& m);

} // namespace model