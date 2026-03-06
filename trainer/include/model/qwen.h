#pragma once
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <string>
#include <cstdint>
#include "model/ffn_expert.h"

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

QwenConfig qwen_config_from_json(const std::string& model_dir);

struct QwenLayer {
    __nv_bfloat16* q_proj = nullptr, *k_proj = nullptr,
                  *v_proj = nullptr, *o_proj = nullptr;
    __nv_bfloat16* q_bias = nullptr, *k_bias = nullptr, *v_bias = nullptr;
    __nv_bfloat16* attn_norm = nullptr, *ffn_norm = nullptr;
    __nv_bfloat16* gate_proj = nullptr, *up_proj = nullptr, *down_proj = nullptr;
};

struct QwenModel {
    QwenConfig cfg;
    static constexpr int MAX_LAYERS = 24;
    __nv_bfloat16* token_embed = nullptr;
    QwenLayer      layers[MAX_LAYERS];
    __nv_bfloat16* final_norm  = nullptr;
    __nv_bfloat16* lm_head     = nullptr;
    bool           lm_head_tied = true;
    float* rope_cos = nullptr, *rope_sin = nullptr;
    float* attn_scratch = nullptr, *qkv_scratch = nullptr, *ffn_scratch = nullptr;
    int max_tokens = 0, max_seq = 0;
};

void qwen_load(QwenModel& m, const std::string& model_dir);
void qwen_free(QwenModel& m);
void qwen_alloc_scratch(QwenModel& m, int max_batch, int max_seq);

void qwen_forward(QwenModel& m, cublasHandle_t cublas,
                  const int32_t* tokens, int batch, int seq,
                  const float* expert_delta, float* logits,
                  __nv_bfloat16* h_states, __nv_bfloat16* h_after_attn,
                  __nv_bfloat16* ffn_norm_out,
                  float* ffn_rms, float* attn_rms, float* final_rms);

void qwen_backward(QwenModel& m, cublasHandle_t cublas,
                   const float* d_logits, int batch, int seq,
                   const __nv_bfloat16* h_states,
                   const __nv_bfloat16* h_after_attn,
                   const __nv_bfloat16* ffn_norm_out,
                   const float* ffn_rms, const float* final_rms,
                   ExpertFFN& expert,
                   const __nv_bfloat16* expert_ffn_norm_out,
                   float* d_h_scratch);

} // namespace model
