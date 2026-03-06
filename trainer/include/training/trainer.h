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

    int32_t* d_tokens  = nullptr;
    int32_t* d_targets = nullptr;
    float*   d_logits  = nullptr;
    float*   d_dlogits = nullptr;

    __nv_bfloat16* h_states            = nullptr;  // [L+1, B*S, H]
    __nv_bfloat16* h_after_attn        = nullptr;  // [L,   B*S, H]
    __nv_bfloat16* ffn_norm_out        = nullptr;  // [L,   B*S, H]  written by pass 2
    __nv_bfloat16* ffn_norm_out_expert = nullptr;  // [L,   B*S, H]  copy from pass 1
    float*          ffn_rms      = nullptr;
    float*          attn_rms     = nullptr;
    float*          final_rms    = nullptr;
    float*          expert_delta = nullptr;
    float*          d_h_scratch  = nullptr;

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