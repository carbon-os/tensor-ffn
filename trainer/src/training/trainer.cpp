#include "training/trainer.h"
#include "training/loss.h"
#include "io/checkpoint.h"
#include "backend/memory.h"
#include "backend/cuda_ops.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cfloat>
#include <cassert>
#include <string>
#include <vector>

namespace training {

// ================================================================
//  Debug helpers
// ================================================================

static float device_l2_norm(const float* d_ptr, int n) {
    std::vector<float> h(n);
    cudaMemcpy(h.data(), d_ptr, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost);
    double acc = 0.0;
    for (int i = 0; i < n; ++i) acc += (double)h[i] * h[i];
    return (float)sqrt(acc);
}

static void print_logit_stats(const float* d_logits, int V) {
    // Sample first token only — cheap
    std::vector<float> h(V);
    cudaMemcpy(h.data(), d_logits, (size_t)V * sizeof(float), cudaMemcpyDeviceToHost);
    float mn = FLT_MAX, mx = -FLT_MAX, sum = 0.f;
    for (int i = 0; i < V; ++i) {
        mn   = fminf(mn, h[i]);
        mx   = fmaxf(mx, h[i]);
        sum += h[i];
    }
    printf("  [logit stats token0] min=%.3f  max=%.3f  mean=%.6f\n",
           mn, mx, sum / (float)V);
}

static float gnorm_expert(const model::ExpertFFN& e) {
    double acc = 0.0;
    for (int l = 0; l < e.num_layers; ++l) {
        const auto& el = e.layers[l];
        int gu = e.intermediate_size * e.hidden_size;
        int d  = e.hidden_size       * e.intermediate_size;

        std::vector<float> tmp(std::max(gu, d));

        cudaMemcpy(tmp.data(), el.gate_grad, gu * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < gu; ++i) acc += (double)tmp[i] * tmp[i];

        cudaMemcpy(tmp.data(), el.up_grad,   gu * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < gu; ++i) acc += (double)tmp[i] * tmp[i];

        cudaMemcpy(tmp.data(), el.down_grad, d  * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < d;  ++i) acc += (double)tmp[i] * tmp[i];
    }
    return (float)sqrt(acc);
}

static void print_weight_stats(const model::ExpertFFN& e, int layer) {
    const auto& el = e.layers[layer];
    int gu = e.intermediate_size * e.hidden_size;

    std::vector<float> tmp(gu);
    cudaMemcpy(tmp.data(), el.gate_proj_fp32, gu * sizeof(float), cudaMemcpyDeviceToHost);

    float mn = FLT_MAX, mx = -FLT_MAX;
    double sum = 0.0;
    for (int i = 0; i < gu; ++i) {
        mn   = fminf(mn, tmp[i]);
        mx   = fmaxf(mx, tmp[i]);
        sum += tmp[i];
    }
    printf("  [layer %d gate_proj_fp32] min=%.4f  max=%.4f  mean=%.6f\n",
           layer, mn, mx, (float)(sum / gu));
}

// ================================================================
//  Auto batch
// ================================================================

int trainer_auto_batch(const model::QwenModel& m, int seq_len) {
    size_t free_bytes, total_bytes;
    cudaMemGetInfo(&free_bytes, &total_bytes);

    auto& c = m.cfg;

    size_t reserved = 1536ULL * 1024 * 1024;
    if (free_bytes <= reserved) return 1;
    size_t budget = free_bytes - reserved;

    size_t S  = (size_t)seq_len;
    size_t H  = (size_t)c.hidden_size;
    size_t I  = (size_t)c.intermediate_size;
    size_t L  = (size_t)c.num_layers;
    size_t nh = (size_t)c.num_heads;
    size_t V  = (size_t)c.vocab_size;

    size_t per_batch =
        nh * S * S * 4
      + (L+1) * S * H * 2
      + L     * S * H * 2
      + L     * S * H * 4
      + L     * S * H * 4 * 2
      +         S * H * 4
      +         S * V * 4 * 2
      +         S * 4 * 2
      + L * I * H * 2 * 3
      + L * I * H * 4 * 3;

    int batch = (int)(budget / per_batch);
    if (batch < 1) batch = 1;
    if (batch > 8) batch = 8;

    printf("VRAM budget: %.1fGB free, %.1fGB per batch → batch=%d\n",
           (double)free_bytes  / (1024.0*1024*1024),
           (double)per_batch   / (1024.0*1024*1024),
           batch);

    return batch;
}

// ================================================================
//  Init
// ================================================================

void trainer_init(TrainContext& ctx, const TrainConfig& cfg) {
    CUBLAS_CHECK(cublasCreate(&ctx.cublas));
    CUBLAS_CHECK(cublasSetMathMode(ctx.cublas, CUBLAS_TF32_TENSOR_OP_MATH));

    printf("Loading model from %s...\n", cfg.model_dir.c_str());
    model::qwen_load(ctx.model, cfg.model_dir);
    auto& c = ctx.model.cfg;
    printf("Model loaded. layers=%d hidden=%d intermediate=%d vocab=%d\n",
           c.num_layers, c.hidden_size, c.intermediate_size, c.vocab_size);

    ctx.seq   = std::min(cfg.seq_len, 2048);
    ctx.batch = cfg.batch_size > 0
                    ? cfg.batch_size
                    : trainer_auto_batch(ctx.model, ctx.seq);
    printf("batch=%d seq=%d\n", ctx.batch, ctx.seq);

    model::qwen_alloc_scratch(ctx.model, ctx.batch, ctx.seq);

    printf("Loading corpus from %s...\n", cfg.corpus_path.c_str());
    ctx.corpus.open(cfg.corpus_path);
    printf("Corpus: %llu tokens\n",
           (unsigned long long)ctx.corpus.total_tokens());

    int max_tokens = ctx.batch * ctx.seq;
    model::expert_alloc(ctx.expert,
                        c.num_layers,
                        c.hidden_size,
                        c.intermediate_size,
                        max_tokens);
    model::expert_init_weights(ctx.expert);

    int total_steps = cfg.steps;
    if (total_steps == 0) {
        total_steps = (int)(ctx.corpus.total_tokens() /
                            ((uint64_t)ctx.batch * ctx.seq));
        if (total_steps == 0) total_steps = 1;
    }

    AdamWConfig acfg;
    acfg.lr = cfg.lr;
    int warmup = std::max(10, std::min(200, total_steps / 10));
    adamw_init(ctx.opt, acfg, total_steps, warmup);
    printf("steps=%d lr=%.2e warmup=%d\n",
           total_steps, (double)acfg.lr, warmup);

    size_t T = (size_t)max_tokens;
    ctx.d_tokens  = (int32_t*)backend::device_alloc(T * sizeof(int32_t));
    ctx.d_targets = (int32_t*)backend::device_alloc(T * sizeof(int32_t));
    ctx.d_logits  = (float*)  backend::device_alloc(T * c.vocab_size * sizeof(float));
    ctx.d_dlogits = (float*)  backend::device_alloc(T * c.vocab_size * sizeof(float));

    size_t L = (size_t)c.num_layers;
    size_t H = (size_t)c.hidden_size;

    ctx.h_states            = (__nv_bfloat16*)backend::device_alloc((L+1) * T * H * sizeof(__nv_bfloat16));
    ctx.h_after_attn        = (__nv_bfloat16*)backend::device_alloc(L     * T * H * sizeof(__nv_bfloat16));
    ctx.ffn_norm_out        = (__nv_bfloat16*)backend::device_alloc(L     * T * H * sizeof(__nv_bfloat16));
    ctx.ffn_norm_out_expert = (__nv_bfloat16*)backend::device_alloc(L     * T * H * sizeof(__nv_bfloat16));
    ctx.ffn_rms      = (float*)backend::device_alloc(L * T * sizeof(float));
    ctx.attn_rms     = (float*)backend::device_alloc(L * T * sizeof(float));
    ctx.final_rms    = (float*)backend::device_alloc(    T * sizeof(float));
    ctx.expert_delta = (float*)backend::device_alloc(L * T * H * sizeof(float));
    ctx.d_h_scratch  = (float*)backend::device_alloc(    T * H * sizeof(float));
}

// ================================================================
//  Run
// ================================================================

void trainer_run(TrainContext& ctx, const TrainConfig& cfg) {
    auto& c       = ctx.model.cfg;
    int   T       = ctx.batch * ctx.seq;
    int   V       = c.vocab_size;
    int   total_steps = ctx.opt.total_steps;

    float random_loss = logf((float)V);
    printf("\n=== Training Diagnostics ===\n");
    printf("Random-chance loss ceiling : %.4f  (ln(%d))\n", random_loss, V);
    printf("LR schedule                : warmup=%d  total=%d  base_lr=%.2e\n",
           ctx.opt.warmup_steps, total_steps, (double)ctx.opt.cfg.lr);
    printf("Grad clip (per-layer)      : %.2f\n", (double)ctx.opt.cfg.grad_clip);
    printf("Expert output scale        : %.3f\n", model::kExpertOutputScale);
    if (ctx.opt.warmup_steps >= total_steps)
        printf("WARNING: warmup_steps (%d) >= total_steps (%d)\n",
               ctx.opt.warmup_steps, total_steps);
    printf("============================\n\n");

    std::vector<uint32_t> h_input ((size_t)T);
    std::vector<uint32_t> h_target((size_t)T);
    uint64_t corpus_offset = 0;

    printf("Starting training for %d steps...\n", total_steps);

    for (int step = 0; step < total_steps; ++step) {
        const bool do_log = ((step + 1) % cfg.log_every == 0) || (step == 0);

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

        backend::host_to_device(ctx.d_tokens,  h_input.data(),  (size_t)T * sizeof(int32_t));
        backend::host_to_device(ctx.d_targets, h_target.data(), (size_t)T * sizeof(int32_t));

        model::expert_zero_grad(ctx.expert);

        // ---- Pass 1: base model only — saves h_after_attn and ffn_norm_out ----
        backend::device_memset(ctx.expert_delta, 0,
                               (size_t)c.num_layers * T * c.hidden_size * sizeof(float));

        model::qwen_forward(ctx.model, ctx.cublas,
                            ctx.d_tokens, ctx.batch, ctx.seq,
                            /*expert_delta=*/nullptr,
                            ctx.d_logits,
                            ctx.h_states,
                            ctx.h_after_attn,
                            ctx.ffn_norm_out,
                            ctx.ffn_rms,
                            ctx.attn_rms,
                            ctx.final_rms);

        // Expert forward reads pass-1 ffn_norm_out
        model::expert_forward(ctx.expert, ctx.cublas,
                              ctx.ffn_norm_out,
                              ctx.expert_delta,
                              T);

        // Save pass-1 ffn_norm_out before pass 2 overwrites it.
        // expert_backward_layer needs the activations the expert actually saw.
        backend::device_copy(ctx.ffn_norm_out_expert,
                             ctx.ffn_norm_out,
                             (size_t)c.num_layers * T * c.hidden_size * sizeof(__nv_bfloat16));

        if (do_log) {
            float total_delta_sq = 0.f;
            printf("  [expert_delta norms]\n");
            for (int l = 0; l < c.num_layers; ++l) {
                const float* ed_l = ctx.expert_delta + (size_t)l * T * c.hidden_size;
                float n = device_l2_norm(ed_l, T * c.hidden_size);
                total_delta_sq += n * n;
                printf("    layer %2d: %.6f\n", l, n);
            }
            printf("  [expert_delta total L2]: %.6f\n", sqrtf(total_delta_sq));
        }

        // ---- Pass 2: base + expert delta — overwrites h_states, h_after_attn, ffn_norm_out ----
        // Backward must be consistent with the loss computed here, so we save
        // the correct activations (including the effect of expert_delta on residuals).
        model::qwen_forward(ctx.model, ctx.cublas,
                            ctx.d_tokens, ctx.batch, ctx.seq,
                            ctx.expert_delta,
                            ctx.d_logits,
                            ctx.h_states,
                            ctx.h_after_attn,
                            ctx.ffn_norm_out,
                            ctx.ffn_rms,
                            ctx.attn_rms,
                            ctx.final_rms);

        if (do_log) print_logit_stats(ctx.d_logits, V);

        // ---- Loss ----
        float loss = training::cross_entropy_loss(
            ctx.cublas,
            ctx.d_logits,
            ctx.d_targets,
            ctx.d_dlogits,
            T, V);

        // ---- Backward ----
        // qwen_backward interleaves expert_backward_layer per layer.
        // Passes pass-2 activations (h_states, h_after_attn, ffn_norm_out) for
        // the base FFN Jacobian, and pass-1 ffn_norm_out_expert for the expert
        // weight/input gradients.
        model::qwen_backward(ctx.model, ctx.cublas,
                             ctx.d_dlogits, ctx.batch, ctx.seq,
                             ctx.h_states,
                             ctx.h_after_attn,
                             ctx.ffn_norm_out,
                             ctx.ffn_rms,
                             ctx.final_rms,
                             ctx.expert,
                             ctx.ffn_norm_out_expert,
                             ctx.d_h_scratch);

        if (do_log) {
            float gnorm_pre = gnorm_expert(ctx.expert);
            printf("  [gnorm pre-clip]:  %.4f\n", gnorm_pre);
        }

        float gnorm_returned = adamw_clip_grads(ctx.expert, ctx.opt.cfg.grad_clip);

        if (do_log) {
            float gnorm_post = gnorm_expert(ctx.expert);
            printf("  [gnorm post-clip]: %.4f  (returned %.4f)\n",
                   gnorm_post, gnorm_returned);
        }

        adamw_step(ctx.opt, ctx.expert);

        if (do_log) print_weight_stats(ctx.expert, 0);

        if (do_log) {
            bool in_warmup = (ctx.opt.step <= ctx.opt.warmup_steps);
            printf("step %5d/%d | loss %.4f (rand=%.4f, delta=%.4f) "
                   "| lr %.2e | gnorm %.3f | warmup=%s\n\n",
                   step + 1, total_steps,
                   loss, random_loss, loss - random_loss,
                   (double)adamw_lr(ctx.opt),
                   gnorm_returned,
                   in_warmup ? "YES" : "NO");
        }

        if (cfg.save_every > 0 && (step + 1) % cfg.save_every == 0) {
            std::string ckpt = cfg.out_path + ".step" + std::to_string(step + 1);
            io::checkpoint_save(ckpt, ctx.expert);
        }
    }

    io::checkpoint_save(cfg.out_path, ctx.expert);
    printf("Training complete. Expert saved to %s\n", cfg.out_path.c_str());
}

// ================================================================
//  Free
// ================================================================

void trainer_free(TrainContext& ctx) {
    model::qwen_free(ctx.model);
    model::expert_free(ctx.expert);
    ctx.corpus.close();

    backend::device_free(ctx.d_tokens);
    backend::device_free(ctx.d_targets);
    backend::device_free(ctx.d_logits);
    backend::device_free(ctx.d_dlogits);
    backend::device_free(ctx.h_states);
    backend::device_free(ctx.h_after_attn);
    backend::device_free(ctx.ffn_norm_out);
    backend::device_free(ctx.ffn_norm_out_expert);
    backend::device_free(ctx.ffn_rms);
    backend::device_free(ctx.attn_rms);
    backend::device_free(ctx.final_rms);
    backend::device_free(ctx.expert_delta);
    backend::device_free(ctx.d_h_scratch);

    if (ctx.cublas) {
        cublasDestroy(ctx.cublas);
        ctx.cublas = nullptr;
    }
}

} // namespace training