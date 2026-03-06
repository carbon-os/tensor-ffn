// ============================================================
// FILE: trainer/include/training/trainer.h
// ============================================================
#pragma once
#include "model/qwen.h"
#include "model/ffn_expert.h"
#include "io/corpus.h"
#include "training/optimizer.h"
#include <cublas_v2.h>
#include <string>

namespace training {

struct TrainConfig {
    std::string model_dir;
    std::string corpus_path;
    std::string out_path;

    int   batch_size  = 0;     // 0 = auto-tune from VRAM
    int   seq_len     = 2048;  // capped from config; override with --seq
    int   steps       = 0;     // 0 = one epoch (full corpus pass)
    float lr          = 3e-4f;
    int   log_every   = 10;    // log loss every N steps
    int   save_every  = 1000;  // checkpoint every N steps
};

struct TrainContext {
    model::QwenModel   model;
    model::ExpertFFN   expert;
    io::CorpusLoader   corpus;
    AdamWState         opt;
    cublasHandle_t     cublas = nullptr;

    // Batch buffers on device
    int32_t* d_tokens  = nullptr;  // [B, S]
    int32_t* d_targets = nullptr;  // [B*S]
    float*   d_logits  = nullptr;  // [B*S, V]
    float*   d_dlogits = nullptr;  // [B*S, V]

    // Activation buffers (reused every step)
    __nv_bfloat16* h_states     = nullptr;  // [L+1, B*S, H]
    __nv_bfloat16* ffn_norm_out = nullptr;  // [L, B*S, H]
    float*          ffn_rms      = nullptr;  // [L, B*S]
    float*          attn_rms     = nullptr;  // [L, B*S]
    float*          final_rms    = nullptr;  // [B*S]
    float*          expert_delta = nullptr;  // [L, B*S, H]
    float*          d_delta      = nullptr;  // [L, B*S, H]
    float*          d_xnorm      = nullptr;  // [L, B*S, H]
    float*          d_h_scratch  = nullptr;  // [B*S, H]

    int batch = 0;
    int seq   = 0;
};

// Initialise training context.
// Loads model + corpus, auto-tunes batch size if cfg.batch_size == 0.
void trainer_init(TrainContext& ctx, const TrainConfig& cfg);

// Run training loop for cfg.steps steps (or full epoch if 0).
void trainer_run(TrainContext& ctx, const TrainConfig& cfg);

// Free all resources.
void trainer_free(TrainContext& ctx);

// Auto-tune batch size: fill available VRAM after model and buffers.
int  trainer_auto_batch(const model::QwenModel& m, int seq_len);

} // namespace training