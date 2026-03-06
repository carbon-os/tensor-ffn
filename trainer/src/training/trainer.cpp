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
#include <string>
#include <vector>

namespace training {

int trainer_auto_batch(const model::QwenModel& m, int seq_len) {
    size_t free_bytes, total_bytes;
    cudaMemGetInfo(&free_bytes, &total_bytes);

    auto& c = m.cfg;

    // Reserve 1.5GB headroom for model weights + misc
    size_t reserved = 1536ULL * 1024 * 1024;
    if (free_bytes <= reserved) return 1;
    size_t budget = free_bytes - reserved;

    // Per-batch memory cost (bytes):
    //   attention scratch: nh * seq * seq * 4
    //   h_states:         (L+1) * seq * H * 2
    //   ffn_norm_out:      L    * seq * H * 2
    //   expert_delta:      L    * seq * H * 4
    //   d_delta + d_xnorm: L   * seq * H * 4 * 2
    //   d_h_scratch:             seq * H * 4
    //   logits:                  seq * V * 4 * 2  (logits + dlogits)
    //   tokens + targets:        seq * 4 * 2
    size_t S  = (size_t)seq_len;
    size_t H  = (size_t)c.hidden_size;
    size_t I  = (size_t)c.intermediate_size;
    size_t L  = (size_t)c.num_layers;
    size_t nh = (size_t)c.num_heads;
    size_t V  = (size_t)c.vocab_size;

    size_t per_batch =
        nh * S * S * 4                  // attn scratch
      + (L+1) * S * H * 2              // h_states
      + L     * S * H * 2              // ffn_norm_out
      + L     * S * H * 4              // expert_delta
      + L     * S * H * 4 * 2         // d_delta + d_xnorm
      +         S * H * 4              // d_h_scratch
      +         S * V * 4 * 2          // logits + dlogits
      +         S * 4 * 2              // tokens + targets
      + L * I * H * 2 * 3             // expert gate/up/act bufs
      + L * I * H * 4 * 3;            // expert fp32 grad bufs

    int batch = (int)(budget / per_batch);
    if (batch < 1) batch = 1;
    if (batch > 8) batch = 8;   // cap — diminishing returns beyond 8

    printf("VRAM budget: %.1fGB free, %.1fGB per batch → batch=%d\n",
           (double)free_bytes / (1024*1024*1024),
           (double)per_batch  / (1024*1024*1024),
           batch);

    return batch;
}

void trainer_init(TrainContext& ctx, const TrainConfig& cfg) {
    // cuBLAS
    CUBLAS_CHECK(cublasCreate(&ctx.cublas));
    CUBLAS_CHECK(cublasSetMathMode(ctx.cublas, CUBLAS_TF32_TENSOR_OP_MATH));

    // Load model
    printf("Loading model from %s...\n", cfg.model_dir.c_str());
    model::qwen_load(ctx.model, cfg.model_dir);
    auto& c = ctx.model.cfg;
    printf("Model loaded. layers=%d hidden=%d intermediate=%d vocab=%d\n",
           c.num_layers, c.hidden_size, c.intermediate_size, c.vocab_size);

    // Batch / seq
    ctx.seq   = std::min(cfg.seq_len, 2048);
    ctx.batch = cfg.batch_size > 0
                    ? cfg.batch_size
                    : trainer_auto_batch(ctx.model, ctx.seq);
    printf("batch=%d seq=%d\n", ctx.batch, ctx.seq);

    model::qwen_alloc_scratch(ctx.model, ctx.batch, ctx.seq);

    // Corpus
    printf("Loading corpus from %s...\n", cfg.corpus_path.c_str());
    ctx.corpus.open(cfg.corpus_path);
    printf("Corpus: %llu tokens\n",
           (unsigned long long)ctx.corpus.total_tokens());

    // Expert
    int max_tokens = ctx.batch * ctx.seq;
    model::expert_alloc(ctx.expert,
                        c.num_layers,
                        c.hidden_size,
                        c.intermediate_size,
                        max_tokens);
    model::expert_init_weights(ctx.expert);

    // Optimizer
    int total_steps = cfg.steps;
    if (total_steps == 0) {
        total_steps = (int)(ctx.corpus.total_tokens() /
                            ((uint64_t)ctx.batch * ctx.seq));
        if (total_steps == 0) total_steps = 1;
    }
    AdamWConfig acfg;
    acfg.lr = cfg.lr;
    adamw_init(ctx.opt, acfg, total_steps);
    printf("steps=%d lr=%.2e\n", total_steps, (double)acfg.lr);

    // Device batch buffers
    size_t T = (size_t)max_tokens;
    ctx.d_tokens  = (int32_t*)backend::device_alloc(T * sizeof(int32_t));
    ctx.d_targets = (int32_t*)backend::device_alloc(T * sizeof(int32_t));
    ctx.d_logits  = (float*)  backend::device_alloc(T * c.vocab_size * sizeof(float));
    ctx.d_dlogits = (float*)  backend::device_alloc(T * c.vocab_size * sizeof(float));

    // Activation buffers
    size_t L = (size_t)c.num_layers;
    size_t H = (size_t)c.hidden_size;

    ctx.h_states     = (__nv_bfloat16*)backend::device_alloc((L+1) * T * H * sizeof(__nv_bfloat16));
    ctx.ffn_norm_out = (__nv_bfloat16*)backend::device_alloc(L     * T * H * sizeof(__nv_bfloat16));
    ctx.ffn_rms      = (float*)        backend::device_alloc(L * T * sizeof(float));
    ctx.attn_rms     = (float*)        backend::device_alloc(L * T * sizeof(float));
    ctx.final_rms    = (float*)        backend::device_alloc(    T * sizeof(float));
    ctx.expert_delta = (float*)        backend::device_alloc(L * T * H * sizeof(float));
    ctx.d_delta      = (float*)        backend::device_alloc(L * T * H * sizeof(float));
    ctx.d_xnorm      = (float*)        backend::device_alloc(L * T * H * sizeof(float));
    ctx.d_h_scratch  = (float*)        backend::device_alloc(    T * H * sizeof(float));
}

void trainer_run(TrainContext& ctx, const TrainConfig& cfg) {
    auto& c       = ctx.model.cfg;
    int   T       = ctx.batch * ctx.seq;
    int   V       = c.vocab_size;
    int   total_steps = ctx.opt.total_steps;

    std::vector<uint32_t> h_input ((size_t)T);
    std::vector<uint32_t> h_target((size_t)T);
    uint64_t corpus_offset = 0;

    printf("Starting training for %d steps...\n", total_steps);

    for (int step = 0; step < total_steps; ++step) {

        // ---- Fetch batch ----
        bool ok = ctx.corpus.next_batch(
            h_input.data(), h_target.data(),
            ctx.batch, ctx.seq, corpus_offset);
        if (!ok) {
            printf("Corpus exhausted at step %d, wrapping around.\n", step);
            corpus_offset = 0;
            ok = ctx.corpus.next_batch(
                h_input.data(), h_target.data(),
                ctx.batch, ctx.seq, corpus_offset);
            if (!ok) {
                fprintf(stderr, "Corpus too small for one batch. Aborting.\n");
                break;
            }
        }

        // Upload to device
        backend::host_to_device(ctx.d_tokens,
                                h_input.data(),
                                (size_t)T * sizeof(int32_t));
        backend::host_to_device(ctx.d_targets,
                                h_target.data(),
                                (size_t)T * sizeof(int32_t));

        // Zero expert grads
        model::expert_zero_grad(ctx.expert);

        // ---- Forward pass 1: base model only ----
        // Captures h_states and ffn_norm_out needed by expert forward + backward.
        backend::device_memset(ctx.expert_delta, 0,
                               (size_t)c.num_layers * T * c.hidden_size * sizeof(float));

        model::qwen_forward(ctx.model, ctx.cublas,
                            ctx.d_tokens, ctx.batch, ctx.seq,
                            nullptr,           // no expert delta yet
                            ctx.d_logits,
                            ctx.h_states,
                            ctx.ffn_norm_out,
                            ctx.ffn_rms,
                            ctx.attn_rms,
                            ctx.final_rms);

        // ---- Expert forward ----
        // Uses ffn_norm_out from pass 1 to compute per-layer expert deltas.
        model::expert_forward(ctx.expert, ctx.cublas,
                              ctx.ffn_norm_out,
                              ctx.expert_delta,
                              T);

        // ---- Forward pass 2: base model + expert delta ----
        // Produces correct logits for loss computation.
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
        float loss = training::cross_entropy_loss(
            ctx.cublas,
            ctx.d_logits,
            ctx.d_targets,
            ctx.d_dlogits,
            T, V);

        // ---- Backward ----
        // Zero d_xnorm before backward so expert contribution starts clean.
        backend::device_memset(ctx.d_xnorm, 0,
                               (size_t)c.num_layers * T * c.hidden_size * sizeof(float));

        // Qwen backward: propagates d_logits through frozen model,
        // returns d_delta [L, T, H] into ctx.d_delta for the expert.
        model::qwen_backward(ctx.model, ctx.cublas,
                             ctx.d_dlogits, ctx.batch, ctx.seq,
                             ctx.h_states,
                             ctx.ffn_norm_out,
                             ctx.ffn_rms,
                             ctx.attn_rms,
                             ctx.final_rms,
                             ctx.d_xnorm,
                             ctx.d_h_scratch);

        // Expert backward: accumulates weight gradients,
        // produces d_xnorm [L, T, H] for completeness.
        model::expert_backward(ctx.expert, ctx.cublas,
                               ctx.ffn_norm_out,
                               ctx.d_delta,
                               ctx.d_xnorm,
                               T);

        // ---- Optimizer step ----
        float gnorm = adamw_clip_grads(ctx.expert,
                                       ctx.opt.cfg.grad_clip);
        adamw_step(ctx.opt, ctx.expert);

        // ---- Logging ----
        if ((step + 1) % cfg.log_every == 0) {
            printf("step %5d/%d | loss %.4f | lr %.2e | gnorm %.3f\n",
                   step + 1, total_steps,
                   loss,
                   (double)adamw_lr(ctx.opt),
                   gnorm);
        }

        // ---- Checkpoint ----
        if (cfg.save_every > 0 && (step + 1) % cfg.save_every == 0) {
            std::string ckpt = cfg.out_path
                             + ".step"
                             + std::to_string(step + 1);
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

    if (ctx.cublas) {
        cublasDestroy(ctx.cublas);
        ctx.cublas = nullptr;
    }
}

} // namespace training