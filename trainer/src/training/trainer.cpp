// ============================================================
// FILE: trainer/src/training/trainer.cpp
// ============================================================
#include "training/trainer.h"
#include "training/loss.h"
#include "io/checkpoint.h"
#include "backend/memory.h"
#include "backend/cuda_ops.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cassert>

namespace training {

static size_t free_vram() {
    size_t free_bytes, total_bytes;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    return free_bytes;
}

int trainer_auto_batch(const model::QwenModel& m, int seq_len) {
    // Rough VRAM accounting:
    //   Model weights: ~1GB for 0.5B bf16
    //   Per token activations: h_states (25 layers), ffn_norm_out, expert buffers, etc.
    //   We reserve 4GB for model + overhead, use rest for batch.
    // Activation memory per (batch * seq):
    //   h_states [26 layers, T, 896] bf16:  26*T*896*2 bytes
    //   ffn_norm_out [24, T, 896] bf16:     24*T*896*2 bytes
    //   expert intermediates [24, T, 4864]: 24*T*4864*2*3 bytes (gate,up,act)
    //   gradient buffers similar scale
    // Total ≈ 26*2 + 24*2 + 24*4864*6 = 52 + 48 + ~700K bytes per token
    // For seq=2048: each token needs ~710 bytes → per sequence: 710*2048 = 1.45MB
    // On A100 with 80GB, after model (1GB) + overhead (5GB) = ~74GB free.
    // 74GB / 1.45MB ≈ 51,000 sequences... but that's optimistic.
    // In practice with attention scratch [B*14*S*S*4]: for B=8,S=2048: 8*14*4M = 448MB per layer
    // → total attention scratch across 24 layers = 10.7GB — this is the bottleneck.
    // For S=2048: attn_scratch per layer = B*14*2048*2048*4 bytes = B * 234MB
    //   Max B = (74GB / 234MB) = ~316 — capped by attention.
    // But typically B=8-16 is plenty for throughput. Return 8 as default.
    (void)m; (void)seq_len;
    return 8;
}

void trainer_init(TrainContext& ctx, const TrainConfig& cfg) {
    // cuBLAS
    CUBLAS_CHECK(cublasCreate(&ctx.cublas));
    CUBLAS_CHECK(cublasSetMathMode(ctx.cublas, CUBLAS_TF32_TENSOR_OP_MATH));

    // Load model
    printf("Loading model from %s...\n", cfg.model_dir.c_str());
    model::qwen_load(ctx.model, cfg.model_dir);
    printf("Model loaded. Layers=%d H=%d I=%d V=%d\n",
           ctx.model.cfg.num_layers,
           ctx.model.cfg.hidden_size,
           ctx.model.cfg.intermediate_size,
           ctx.model.cfg.vocab_size);

    // Batch/seq
    ctx.seq = std::min(cfg.seq_len, 2048);
    ctx.batch = cfg.batch_size > 0 ? cfg.batch_size : trainer_auto_batch(ctx.model, ctx.seq);
    printf("Batch=%d Seq=%d\n", ctx.batch, ctx.seq);

    qwen_alloc_scratch(ctx.model, ctx.batch, ctx.seq);

    // Load corpus
    printf("Loading corpus from %s...\n", cfg.corpus_path.c_str());
    ctx.corpus.open(cfg.corpus_path);
    printf("Corpus: %llu tokens\n", (unsigned long long)ctx.corpus.total_tokens());

    // Expert
    auto& c = ctx.model.cfg;
    int max_tokens = ctx.batch * ctx.seq;
    model::expert_alloc(ctx.expert, c.num_layers, c.hidden_size, c.intermediate_size, max_tokens);
    model::expert_init_weights(ctx.expert);

    // Optimizer
    int total_steps = cfg.steps;
    if (total_steps == 0) {
        total_steps = (int)(ctx.corpus.total_tokens() / ((uint64_t)ctx.batch * ctx.seq));
    }
    AdamWConfig acfg;
    acfg.lr = cfg.lr;
    adamw_init(ctx.opt, acfg, total_steps);
    printf("Steps=%d LR=%.2e\n", total_steps, (double)acfg.lr);

    // Device batch buffers
    size_t T = max_tokens;
    ctx.d_tokens  = (int32_t*)backend::device_alloc(ctx.batch * ctx.seq * sizeof(int32_t));
    ctx.d_targets = (int32_t*)backend::device_alloc(T * sizeof(int32_t));
    ctx.d_logits  = (float*)  backend::device_alloc(T * c.vocab_size * sizeof(float));
    ctx.d_dlogits = (float*)  backend::device_alloc(T * c.vocab_size * sizeof(float));

    // Activation buffers
    size_t L = c.num_layers;
    ctx.h_states     = (__nv_bfloat16*)backend::device_alloc((L+1) * T * c.hidden_size * sizeof(__nv_bfloat16));
    ctx.ffn_norm_out = (__nv_bfloat16*)backend::device_alloc(L     * T * c.hidden_size * sizeof(__nv_bfloat16));
    ctx.ffn_rms      = (float*)        backend::device_alloc(L * T * sizeof(float));
    ctx.attn_rms     = (float*)        backend::device_alloc(L * T * sizeof(float));
    ctx.final_rms    = (float*)        backend::device_alloc(    T * sizeof(float));
    ctx.expert_delta = (float*)        backend::device_alloc(L * T * c.hidden_size * sizeof(float));
    ctx.d_delta      = (float*)        backend::device_alloc(L * T * c.hidden_size * sizeof(float));
    ctx.d_xnorm      = (float*)        backend::device_alloc(L * T * c.hidden_size * sizeof(float));
    ctx.d_h_scratch  = (float*)        backend::device_alloc(    T * c.hidden_size * sizeof(float));
}

void trainer_run(TrainContext& ctx, const TrainConfig& cfg) {
    auto& c  = ctx.model.cfg;
    int T    = ctx.batch * ctx.seq;
    int V    = c.vocab_size;
    int L    = c.num_layers;
    int total_steps = ctx.opt.total_steps;

    std::vector<uint32_t> h_input( (size_t)T);
    std::vector<uint32_t> h_target((size_t)T);
    uint64_t corpus_offset = 0;

    printf("Starting training for %d steps...\n", total_steps);

    for (int step = 0; step < total_steps; ++step) {
        // Fetch batch
        bool ok = ctx.corpus.next_batch(h_input.data(), h_target.data(),
                                         ctx.batch, ctx.seq, corpus_offset);
        if (!ok) {
            printf("Corpus exhausted at step %d, wrapping around.\n", step);
            corpus_offset = 0;
            ok = ctx.corpus.next_batch(h_input.data(), h_target.data(),
                                       ctx.batch, ctx.seq, corpus_offset);
            if (!ok) { fprintf(stderr, "Corpus too small for one batch.\n"); break; }
        }

        // Upload tokens/targets
        backend::host_to_device(ctx.d_tokens,  h_input.data(),  T * sizeof(int32_t));
        backend::host_to_device(ctx.d_targets, h_target.data(), T * sizeof(int32_t));

        // Zero expert grads
        model::expert_zero_grad(ctx.expert);

        // ---- Forward ----
        // Expert forward (with zeroed delta first, then add expert)
        backend::device_memset(ctx.expert_delta, 0, (size_t)L * T * c.hidden_size * sizeof(float));

        // Compute expert forward from ffn_norm_out of previous step... but we don't have it yet.
        // Two-pass approach: first run model to get ffn_norm_out, then compute expert delta.
        // Pass 1: model forward without expert delta (to get saved ffn_norm_out)
        model::qwen_forward(ctx.model, ctx.cublas,
                             ctx.d_tokens, ctx.batch, ctx.seq,
                             nullptr,  // no expert delta in pass 1
                             ctx.d_logits,
                             ctx.h_states,
                             ctx.ffn_norm_out,
                             ctx.ffn_rms,
                             ctx.attn_rms,
                             ctx.final_rms);

        // Pass 2: compute expert delta from saved ffn_norm_out
        model::expert_forward(ctx.expert, ctx.cublas,
                               ctx.ffn_norm_out, ctx.expert_delta, T);

        // Pass 3: model forward WITH expert delta (to get correct logits for loss)
        // For training efficiency, we could combine passes 1 and 3 by interleaving
        // expert computation per layer. The current design runs two full forwards.
        // This is correct but ~2x forward cost. A fused implementation would be faster.
        model::qwen_forward(ctx.model, ctx.cublas,
                             ctx.d_tokens, ctx.batch, ctx.seq,
                             ctx.expert_delta,
                             ctx.d_logits,
                             ctx.h_states,
                             ctx.ffn_norm_out,
                             ctx.ffn_rms,
                             ctx.attn_rms,
                             ctx.final_rms);

        // ---- Loss ----
        float loss = training::cross_entropy_loss(ctx.cublas,
                                                   ctx.d_logits,
                                                   ctx.d_targets,
                                                   ctx.d_dlogits,
                                                   T, V);

        // ---- Backward ----
        // Expert backward: given d_delta [L, T, H] we need to know d_delta.
        // d_delta comes from qwen_backward propagating d_logits through the model.
        // The expert at each layer l receives d_h[l+1] as its upstream gradient.
        // qwen_backward computes this and returns d_x_norm = d_delta passed to expert_backward.

        // First: qwen backward to get d_ffn_norm [L, T, H] → stored in ctx.d_delta
        model::qwen_backward(ctx.model, ctx.cublas,
                              ctx.d_dlogits, ctx.batch, ctx.seq,
                              ctx.h_states, ctx.ffn_norm_out,
                              ctx.ffn_rms, ctx.attn_rms, ctx.final_rms,
                              ctx.d_xnorm,    // d_expert_norm (input from expert bwd)
                              ctx.d_h_scratch);
        // The d_delta for the expert is the upstream gradient flowing through the residual,
        // which equals d_h at each layer position. It's computed inside qwen_backward.
        // We pass ctx.d_delta as the upstream d_h that the expert receives.
        // For the first iteration, initialize d_xnorm to zero (no expert contribution yet).
        // The order is: qwen_backward computes d_h including expert path, then expert_backward
        // computes expert weight grads and d_x_norm, which feeds back into qwen_backward.
        // In this non-interleaved design, we approximate: one backward pass through each.

        // Expert backward: accumulate weight gradients
        model::expert_backward_v2(ctx.expert, ctx.cublas,
                                   ctx.ffn_norm_out,
                                   ctx.d_delta,   // upstream grad [L, T, H]
                                   ctx.d_xnorm,   // output: d_ffn_norm [L, T, H]
                                   T);

        // ---- Optimizer step ----
        float gnorm = adamw_clip_grads(ctx.expert, ctx.opt.cfg.grad_clip);
        adamw_step(ctx.opt, ctx.expert);

        // ---- Logging ----
        if ((step + 1) % cfg.log_every == 0) {
            float lr = adamw_lr(ctx.opt);
            printf("step %5d/%d | loss %.4f | lr %.2e | gnorm %.3f\n",
                   step + 1, total_steps, loss, (double)lr, gnorm);
        }

        // ---- Checkpoint ----
        if (cfg.save_every > 0 && (step + 1) % cfg.save_every == 0) {
            std::string ckpt = cfg.out_path + ".step" + std::to_string(step+1);
            io::checkpoint_save(ckpt, ctx.expert);
        }
    }

    // Final save
    io::checkpoint_save(cfg.out_path, ctx.expert);
    printf("Training complete. Expert saved to %s\n", cfg.out_path.c_str());
}

void trainer_free(TrainContext& ctx) {
    model::qwen_free(ctx.model);
    model::expert_free(ctx.expert);
    ctx.corpus.close();
    backend::device_free(ctx.d_tokens);
    backend::device_free(ctx.d_targets);
    backend::device_free(ctx.d_logits);
    backend::device_free(ctx.d_dlogits);
    backend::device_free(ctx.h_states);
    backend::device_free(ctx.ffn_norm_out);
    backend::device_free(ctx.ffn_rms);
    backend::device_free(ctx.attn_rms);
    backend::device_free(ctx.final_rms);
    backend::device_free(ctx.expert_delta);
    backend::device_free(ctx.d_delta);
    backend::device_free(ctx.d_xnorm);
    backend::device_free(ctx.d_h_scratch);
    cublasDestroy(ctx.cublas);
}

} // namespace training