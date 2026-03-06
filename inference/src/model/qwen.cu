#include "model/qwen.h"
#include "model/expert.h"
#include "backend/cuda_ops.h"
#include "backend/memory.h"
#include "io/safetensors.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <cassert>

using json = nlohmann::json;

namespace model {

// ================================================================
//  Config
// ================================================================

QwenConfig qwen_config_from_json(const std::string& dir) {
    std::ifstream f(dir + "/config.json");
    if (!f) { fprintf(stderr, "Cannot open %s/config.json\n", dir.c_str()); exit(1); }
    json j; f >> j;
    QwenConfig c;
    c.hidden_size       = j.value("hidden_size",             896);
    c.intermediate_size = j.value("intermediate_size",      4864);
    c.num_layers        = j.value("num_hidden_layers",         24);
    c.num_heads         = j.value("num_attention_heads",       14);
    c.num_kv_heads      = j.value("num_key_value_heads",        2);
    c.head_dim          = c.hidden_size / c.num_heads;
    c.vocab_size        = j.value("vocab_size",            151936);
    c.max_seq_len       = j.value("max_position_embeddings", 131072);
    c.rms_norm_eps      = j.value("rms_norm_eps",             1e-6f);
    c.rope_theta        = j.value("rope_theta",               1e6f);
    c.tie_embeddings    = j.value("tie_word_embeddings",       true);

    // ---- FIX STARTS HERE ----
    // Handle EOS token (can be int or array)
    if (j.contains("eos_token_id")) {
        if (j["eos_token_id"].is_array() && !j["eos_token_id"].empty()) {
            c.eos_token_id = j["eos_token_id"][0].get<int>();
        } else if (j["eos_token_id"].is_number()) {
            c.eos_token_id = j["eos_token_id"].get<int>();
        }
    } else {
        c.eos_token_id = 151645; // Default
    }

    // Handle BOS token (can be int or array)
    if (j.contains("bos_token_id")) {
        if (j["bos_token_id"].is_array() && !j["bos_token_id"].empty()) {
            c.bos_token_id = j["bos_token_id"][0].get<int>();
        } else if (j["bos_token_id"].is_number()) {
            c.bos_token_id = j["bos_token_id"].get<int>();
        }
    } else {
        c.bos_token_id = 151643; // Default
    }
    // ---- FIX ENDS HERE ----

    return c;
}

// ================================================================
//  Load
// ================================================================

static __nv_bfloat16* load_bf16(const io::SafeTensorsDir& st,
                                  const std::string& name, size_t elems) {
    if (!st.has(name)) { fprintf(stderr, "Missing tensor: %s\n", name.c_str()); exit(1); }
    auto* p = (__nv_bfloat16*)backend::device_alloc(elems * sizeof(__nv_bfloat16));
    size_t got = st.load_to_device(name, p);
    if (got != elems * sizeof(__nv_bfloat16)) {
        fprintf(stderr, "Size mismatch for %s\n", name.c_str()); exit(1);
    }
    return p;
}

void qwen_load(QwenModel& m, const std::string& dir, int max_seq_len) {
    m.cfg = qwen_config_from_json(dir);
    auto& c = m.cfg;

    io::SafeTensorsDir st;
    st.open(dir);

    size_t H = c.hidden_size, I = c.intermediate_size, V = c.vocab_size;
    size_t nh = c.num_heads, nkv = c.num_kv_heads, hd = c.head_dim;

    m.token_embed = load_bf16(st, "model.embed_tokens.weight", V * H);

    for (int l = 0; l < c.num_layers; ++l) {
        auto& ly = m.layers[l];
        std::string pfx = "model.layers." + std::to_string(l) + ".";
        ly.q_proj    = load_bf16(st, pfx+"self_attn.q_proj.weight",   nh*hd * H);
        ly.k_proj    = load_bf16(st, pfx+"self_attn.k_proj.weight",  nkv*hd * H);
        ly.v_proj    = load_bf16(st, pfx+"self_attn.v_proj.weight",  nkv*hd * H);
        ly.o_proj    = load_bf16(st, pfx+"self_attn.o_proj.weight",  H * nh*hd);
        if (st.has(pfx+"self_attn.q_proj.bias")) {
            ly.q_bias = load_bf16(st, pfx+"self_attn.q_proj.bias",   nh*hd);
            ly.k_bias = load_bf16(st, pfx+"self_attn.k_proj.bias",  nkv*hd);
            ly.v_bias = load_bf16(st, pfx+"self_attn.v_proj.bias",  nkv*hd);
        }
        ly.attn_norm = load_bf16(st, pfx+"input_layernorm.weight",            H);
        ly.ffn_norm  = load_bf16(st, pfx+"post_attention_layernorm.weight",   H);
        ly.gate_proj = load_bf16(st, pfx+"mlp.gate_proj.weight", I * H);
        ly.up_proj   = load_bf16(st, pfx+"mlp.up_proj.weight",   I * H);
        ly.down_proj = load_bf16(st, pfx+"mlp.down_proj.weight", H * I);
    }

    m.final_norm = load_bf16(st, "model.norm.weight", H);
    if (c.tie_embeddings) {
        m.lm_head      = m.token_embed;
        m.lm_head_tied = true;
    } else {
        m.lm_head      = load_bf16(st, "lm_head.weight", V * H);
        m.lm_head_tied = false;
    }

    // RoPE tables
    int half_hd = (int)hd / 2;
    std::vector<float> cos_h(max_seq_len * half_hd), sin_h(max_seq_len * half_hd);
    for (int pos = 0; pos < max_seq_len; ++pos)
        for (int i = 0; i < half_hd; ++i) {
            float freq = 1.f / powf(c.rope_theta, (float)(2*i) / (float)hd);
            cos_h[pos*half_hd+i] = cosf(pos*freq);
            sin_h[pos*half_hd+i] = sinf(pos*freq);
        }
    m.rope_cos = (float*)backend::device_alloc(max_seq_len * half_hd * sizeof(float));
    m.rope_sin = (float*)backend::device_alloc(max_seq_len * half_hd * sizeof(float));
    backend::host_to_device(m.rope_cos, cos_h.data(), max_seq_len * half_hd * sizeof(float));
    backend::host_to_device(m.rope_sin, sin_h.data(), max_seq_len * half_hd * sizeof(float));

    // KV cache
    m.kv.num_layers = c.num_layers;
    m.kv.max_len    = max_seq_len;
    m.kv.nkv_heads  = (int)nkv;
    m.kv.head_dim   = (int)hd;
    m.kv.len        = 0;
    size_t kv_elems = (size_t)c.num_layers * max_seq_len * nkv * hd;
    m.kv.k = (__nv_bfloat16*)backend::device_alloc(kv_elems * sizeof(__nv_bfloat16));
    m.kv.v = (__nv_bfloat16*)backend::device_alloc(kv_elems * sizeof(__nv_bfloat16));

    // Scratch buffers
    int   mT  = max_seq_len;
    int   mTL = max_seq_len; // max total_len = max_seq_len
    m.max_T     = mT;
    m.max_total = mTL;

    m.h_work    = (__nv_bfloat16*)backend::device_alloc((size_t)mT * H  * sizeof(__nv_bfloat16));
    m.h_norm    = (__nv_bfloat16*)backend::device_alloc((size_t)mT * H  * sizeof(__nv_bfloat16));
    m.q_bf16    = (__nv_bfloat16*)backend::device_alloc((size_t)mT * nh*hd  * sizeof(__nv_bfloat16));
    m.k_bf16    = (__nv_bfloat16*)backend::device_alloc((size_t)mT * nkv*hd * sizeof(__nv_bfloat16));
    m.v_bf16    = (__nv_bfloat16*)backend::device_alloc((size_t)mT * nkv*hd * sizeof(__nv_bfloat16));
    m.q_fp32    = (float*)backend::device_alloc((size_t)mT  * nh*hd  * sizeof(float));
    m.k_fp32    = (float*)backend::device_alloc((size_t)mT  * nkv*hd * sizeof(float));
    m.k_fp32_ex = (float*)backend::device_alloc((size_t)mTL * nh*hd  * sizeof(float));
    m.v_fp32_ex = (float*)backend::device_alloc((size_t)mTL * nh*hd  * sizeof(float));
    m.q_perm    = (float*)backend::device_alloc((size_t)nh  * mT  * hd * sizeof(float));
    m.k_perm    = (float*)backend::device_alloc((size_t)nh  * mTL * hd * sizeof(float));
    m.v_perm    = (float*)backend::device_alloc((size_t)nh  * mTL * hd * sizeof(float));
    m.scores    = (float*)backend::device_alloc((size_t)nh  * mT * mTL * sizeof(float));
    m.ctx_perm  = (float*)backend::device_alloc((size_t)nh  * mT  * hd * sizeof(float));
    m.ctx_fp32  = (float*)backend::device_alloc((size_t)mT  * H        * sizeof(float));
    m.ctx_bf16  = (__nv_bfloat16*)backend::device_alloc((size_t)mT * H * sizeof(__nv_bfloat16));
    m.gate_buf  = (__nv_bfloat16*)backend::device_alloc((size_t)mT * I * sizeof(__nv_bfloat16));
    m.up_buf    = (__nv_bfloat16*)backend::device_alloc((size_t)mT * I * sizeof(__nv_bfloat16));
    m.act_buf   = (__nv_bfloat16*)backend::device_alloc((size_t)mT * I * sizeof(__nv_bfloat16));
    m.ffn_fp32  = (float*)backend::device_alloc((size_t)mT * H * sizeof(float));
    m.rms_buf   = (float*)backend::device_alloc((size_t)mT      * sizeof(float));
    m.expert_delta = (float*)backend::device_alloc((size_t)mT * H * sizeof(float));
}

void qwen_reset(QwenModel& m) { m.kv.len = 0; }

void qwen_free(QwenModel& m) {
    auto& c = m.cfg;
    backend::device_free(m.token_embed);
    for (int l = 0; l < c.num_layers; ++l) {
        auto& ly = m.layers[l];
        backend::device_free(ly.q_proj); backend::device_free(ly.k_proj);
        backend::device_free(ly.v_proj); backend::device_free(ly.o_proj);
        if (ly.q_bias) {
            backend::device_free(ly.q_bias);
            backend::device_free(ly.k_bias);
            backend::device_free(ly.v_bias);
        }
        backend::device_free(ly.attn_norm); backend::device_free(ly.ffn_norm);
        backend::device_free(ly.gate_proj); backend::device_free(ly.up_proj);
        backend::device_free(ly.down_proj);
    }
    backend::device_free(m.final_norm);
    if (!m.lm_head_tied) backend::device_free(m.lm_head);
    backend::device_free(m.rope_cos);
    backend::device_free(m.rope_sin);
    backend::device_free(m.kv.k);
    backend::device_free(m.kv.v);
    backend::device_free(m.h_work);    backend::device_free(m.h_norm);
    backend::device_free(m.q_bf16);   backend::device_free(m.k_bf16);
    backend::device_free(m.v_bf16);   backend::device_free(m.q_fp32);
    backend::device_free(m.k_fp32);   backend::device_free(m.k_fp32_ex);
    backend::device_free(m.v_fp32_ex);backend::device_free(m.q_perm);
    backend::device_free(m.k_perm);   backend::device_free(m.v_perm);
    backend::device_free(m.scores);   backend::device_free(m.ctx_perm);
    backend::device_free(m.ctx_fp32); backend::device_free(m.ctx_bf16);
    backend::device_free(m.gate_buf); backend::device_free(m.up_buf);
    backend::device_free(m.act_buf);  backend::device_free(m.ffn_fp32);
    backend::device_free(m.rms_buf);  backend::device_free(m.expert_delta);
}

// ================================================================
//  Kernels
// ================================================================

// Token embedding
__global__ static void k_embed(const int32_t* tok, const __nv_bfloat16* tbl,
                                __nv_bfloat16* out, int T, int H) {
    int t = blockIdx.x, h = threadIdx.x + blockIdx.y * blockDim.x;
    if (h < H) out[t*H+h] = tbl[(int)tok[t]*H+h];
}

// Add QKV bias (bf16, in-place)
__global__ static void k_add_bias(__nv_bfloat16* x, const __nv_bfloat16* b, int T, int D) {
    int t = blockIdx.x, d = threadIdx.x + blockIdx.y * blockDim.x;
    if (d < D) x[t*D+d] = __hadd(x[t*D+d], b[d]);
}

// RoPE with position offset.
// x: [T, nh, hd] fp32 in-place.
// cos/sin: [max_seq, hd/2].
// Each new token at local index t has global position (past_len + t).
__global__ static void k_rope(float* x, const float* cos_t, const float* sin_t,
                               int T, int nh, int hd, int past_len) {
    int t = blockIdx.x, h = blockIdx.y, i = threadIdx.x;
    if (i >= hd/2) return;
    int pos  = past_len + t;
    int half = hd / 2;
    float* xh = x + t*nh*hd + h*hd;
    float c = cos_t[pos*half + i];
    float s = sin_t[pos*half + i];
    float x0 = xh[i], x1 = xh[i+half];
    xh[i]      = x0*c - x1*s;
    xh[i+half] = x0*s + x1*c;
}

// Expand KV: read bf16 from cache [total_len, nkv, hd],
// write fp32 output [total_len, nh, hd] with GQA head replication.
__global__ static void k_expand_kv(const __nv_bfloat16* kv_in, float* kv_out,
                                    int total_len, int nkv, int nh, int hd) {
    int t  = blockIdx.x, qh = blockIdx.y, d = threadIdx.x;
    if (d >= hd) return;
    int kv_h = qh * nkv / nh;
    kv_out[t*nh*hd + qh*hd + d] = __bfloat162float(kv_in[t*nkv*hd + kv_h*hd + d]);
}

// Permute [T, nh, hd] → [nh, T, hd]
__global__ static void k_perm_thd_htd(const float* src, float* dst, int T, int nh, int hd) {
    int t = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    if (d >= hd) return;
    dst[h*T*hd + t*hd + d] = src[t*nh*hd + h*hd + d];
}

// Permute [nh, T, hd] → [T, H]  (merge heads, write flat)
__global__ static void k_perm_htd_thd(const float* src, float* dst, int T, int nh, int hd) {
    int h = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    if (d >= hd) return;
    dst[t*nh*hd + h*hd + d] = src[h*T*hd + t*hd + d];
}

// Causal mask for KV-cache attention.
// scores: [nh, T, total_len].  Token at local index t_local (global pos = past_len+t_local)
// can attend to global positions 0..past_len+t_local.
__global__ static void k_causal_mask(float* scores, int nh, int T, int total_len, int past_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nh * T * total_len) return;
    int t = (idx / total_len) % T;
    int k = idx % total_len;
    if (k > past_len + t) scores[idx] = -1e30f;
}

// Residual: h[bf16] += delta[fp32]
__global__ static void k_add_fp32_to_bf16(const float* d, __nv_bfloat16* h, int n) {
    int i = blockIdx.x*256+threadIdx.x;
    if (i < n) h[i] = __float2bfloat16(__bfloat162float(h[i]) + d[i]);
}

// ================================================================
//  Forward pass
// ================================================================

void qwen_forward(QwenModel& m, cublasHandle_t cublas,
                  const int32_t* d_tokens, int T,
                  const ExpertModel* expert,
                  float* d_logits) {
    assert(T > 0);
    assert(m.kv.len + T <= m.kv.max_len && "KV cache overflow — increase --max-seq");

    auto& c   = m.cfg;
    int H     = c.hidden_size, I = c.intermediate_size;
    int nh    = c.num_heads, nkv = c.num_kv_heads, hd = c.head_dim;
    int past  = m.kv.len;      // tokens already in cache
    int total = past + T;      // after this call

    // 1. Embed
    {
        dim3 g(T, (H+255)/256);
        k_embed<<<g, 256>>>(d_tokens, m.token_embed, m.h_work, T, H);
    }

    for (int l = 0; l < c.num_layers; ++l) {
        auto& ly = m.layers[l];

        // ---- Attention ----
        backend::rmsnorm_bf16(m.h_work, ly.attn_norm, m.h_norm, m.rms_buf,
                              T, H, c.rms_norm_eps);

        backend::gemm_bf16_out(cublas, T, nh*hd,  H, m.h_norm, ly.q_proj, m.q_bf16, false, true);
        backend::gemm_bf16_out(cublas, T, nkv*hd, H, m.h_norm, ly.k_proj, m.k_bf16, false, true);
        backend::gemm_bf16_out(cublas, T, nkv*hd, H, m.h_norm, ly.v_proj, m.v_bf16, false, true);

        if (ly.q_bias) {
            dim3 gq(T, (nh*hd+255)/256), gkv(T, (nkv*hd+255)/256);
            k_add_bias<<<gq,  256>>>(m.q_bf16, ly.q_bias, T, nh*hd);
            k_add_bias<<<gkv, 256>>>(m.k_bf16, ly.k_bias, T, nkv*hd);
            k_add_bias<<<gkv, 256>>>(m.v_bf16, ly.v_bias, T, nkv*hd);
        }

        // Cast Q, K to fp32 for RoPE
        backend::cast_bf16_to_fp32(m.q_bf16, m.q_fp32, T * nh  * hd);
        backend::cast_bf16_to_fp32(m.k_bf16, m.k_fp32, T * nkv * hd);

        // RoPE (positions past..past+T-1)
        {
            dim3 gq(T, nh), gkv(T, nkv);
            k_rope<<<gq,  hd/2>>>(m.q_fp32, m.rope_cos, m.rope_sin, T, nh,  hd, past);
            k_rope<<<gkv, hd/2>>>(m.k_fp32, m.rope_cos, m.rope_sin, T, nkv, hd, past);
        }

        // Write new K, V into cache
        {
            __nv_bfloat16* kc = m.kv.k_layer(l) + (size_t)past * nkv * hd;
            __nv_bfloat16* vc = m.kv.v_layer(l) + (size_t)past * nkv * hd;
            backend::cast_fp32_to_bf16(m.k_fp32, kc, T * nkv * hd);
            // V was never cast to fp32 — write bf16 directly
            backend::device_copy(vc, m.v_bf16, (size_t)T * nkv * hd * sizeof(__nv_bfloat16));
        }

        // Expand K, V from cache: [total, nkv, hd] bf16 → [total, nh, hd] fp32 (GQA)
        {
            dim3 g(total, nh);
            k_expand_kv<<<g, hd>>>(m.kv.k_layer(l), m.k_fp32_ex, total, nkv, nh, hd);
            k_expand_kv<<<g, hd>>>(m.kv.v_layer(l), m.v_fp32_ex, total, nkv, nh, hd);
        }

        // Permute Q [T, nh, hd] → q_perm [nh, T, hd]
        {
            dim3 g(T, nh);
            k_perm_thd_htd<<<g, hd>>>(m.q_fp32, m.q_perm, T, nh, hd);
        }
        // Permute K [total, nh, hd] → k_perm [nh, total, hd]
        // Permute V [total, nh, hd] → v_perm [nh, total, hd]
        {
            dim3 g(total, nh);
            k_perm_thd_htd<<<g, hd>>>(m.k_fp32_ex, m.k_perm, total, nh, hd);
            k_perm_thd_htd<<<g, hd>>>(m.v_fp32_ex, m.v_perm, total, nh, hd);
        }

        // scores [nh, T, total] = q_perm @ k_perm^T / sqrt(hd)
        {
            float scale = 1.f / sqrtf((float)hd), zero = 0.f;
            CUBLAS_CHECK(cublasSgemmStridedBatched(cublas,
                CUBLAS_OP_T, CUBLAS_OP_N,
                total, T, hd,
                &scale,
                m.k_perm, hd, (long long)total * hd,
                m.q_perm, hd, (long long)T     * hd,
                &zero,
                m.scores, total, (long long)T * total,
                nh));
        }

        // Causal mask (no-op for single decode token when T=1, past=all)
        if (T > 1 || past == 0) {
            int n = nh * T * total;
            k_causal_mask<<<(n+255)/256, 256>>>(m.scores, nh, T, total, past);
        }

        // Softmax over total_len dimension: rows = nh*T
        backend::softmax_fp32(m.scores, nh * T, total);

        // ctx_perm [nh, T, hd] = scores @ v_perm
        {
            float one = 1.f, zero = 0.f;
            CUBLAS_CHECK(cublasSgemmStridedBatched(cublas,
                CUBLAS_OP_N, CUBLAS_OP_N,
                hd, T, total,
                &one,
                m.v_perm,  hd,    (long long)total * hd,
                m.scores,  total, (long long)T     * total,
                &zero,
                m.ctx_perm, hd,   (long long)T * hd,
                nh));
        }

        // Permute ctx_perm [nh, T, hd] → ctx_fp32 [T, H]
        {
            dim3 g(nh, T);
            k_perm_htd_thd<<<g, hd>>>(m.ctx_perm, m.ctx_fp32, T, nh, hd);
        }

        // o_proj: [T, H] → h_work contribution (bf16 out, then add residual)
        backend::cast_fp32_to_bf16(m.ctx_fp32, m.ctx_bf16, T * H);
        backend::gemm_bf16(cublas, T, H, nh*hd,
                           1.f, m.ctx_bf16, ly.o_proj,
                           0.f, m.ffn_fp32, false, true);
        {
            int n = T * H;
            k_add_fp32_to_bf16<<<(n+255)/256, 256>>>(m.ffn_fp32, m.h_work, n);
        }

        // ---- FFN ----
        backend::rmsnorm_bf16(m.h_work, ly.ffn_norm, m.h_norm, m.rms_buf,
                              T, H, c.rms_norm_eps);

        // Expert delta (written to m.expert_delta, then added to base ffn output)
        bool do_expert = (expert != nullptr);
        if (do_expert)
            expert_forward_layer(*const_cast<ExpertModel*>(expert), cublas, l,
                                 m.h_norm, m.expert_delta, T);

        // Base FFN: SwiGLU
        backend::gemm_bf16_out(cublas, T, I, H, m.h_norm, ly.gate_proj, m.gate_buf, false, true);
        backend::gemm_bf16_out(cublas, T, I, H, m.h_norm, ly.up_proj,   m.up_buf,   false, true);
        backend::silu_mul_bf16(m.gate_buf, m.up_buf, m.act_buf, T * I);
        backend::gemm_bf16(cublas, T, H, I,
                           1.f, m.act_buf, ly.down_proj,
                           0.f, m.ffn_fp32, false, true);

        // Add expert delta if present
        if (do_expert)
            backend::add_fp32(m.ffn_fp32, m.expert_delta, m.ffn_fp32, T * H);

        // Residual
        {
            int n = T * H;
            k_add_fp32_to_bf16<<<(n+255)/256, 256>>>(m.ffn_fp32, m.h_work, n);
        }
    }

    // Final norm + lm_head → logits
    backend::rmsnorm_bf16(m.h_work, m.final_norm, m.h_norm, m.rms_buf, T, H, c.rms_norm_eps);
    backend::gemm_bf16(cublas, T, c.vocab_size, H,
                       1.f, m.h_norm, m.lm_head,
                       0.f, d_logits, false, true);

    // Update cache length
    m.kv.len += T;
}

} // namespace model