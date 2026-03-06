#include "backend/cuda_ops.h"
#include "backend/memory.h"
#include "model/qwen.h"
#include "model/expert.h"
#include "io/tokenizer.h"
#include "io/checkpoint.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <random>

static void usage() {
    fprintf(stderr,
        "tensor-ffn-inference\n"
        "\n"
        "  --model        <dir>   HuggingFace model directory  (required)\n"
        "  --prompt       <str>   Input prompt                  (required)\n"
        "  --chat                 Enable ChatML formatting      (optional)\n"
        "  --expert       <path>  Expert .bin file              (optional)\n"
        "  --expert-scale <F>     Expert delta scale            (default 0.1)\n"
        "  --max-tokens   <N>     Tokens to generate            (default 128)\n"
        "  --temperature  <F>     Sampling temperature          (default 0 = greedy)\n"
        "  --max-seq      <N>     KV cache capacity             (default 2048)\n"
        "\n");
    exit(1);
}

static std::string get_flag(int argc, char** argv, const char* flag, const char* def = "") {
    for (int i = 1; i < argc - 1; ++i)
        if (strcmp(argv[i], flag) == 0) return argv[i+1];
    return def;
}
static int   get_flag_i(int argc, char** argv, const char* f, int   d) {
    auto v = get_flag(argc, argv, f); return v.empty() ? d : std::stoi(v);
}
static float get_flag_f(int argc, char** argv, const char* f, float d) {
    auto v = get_flag(argc, argv, f); return v.empty() ? d : std::stof(v);
}

static int sample(const float* logits, int V, float temperature, std::mt19937& rng) {
    if (temperature <= 0.f) {
        int best = 0;
        for (int i = 1; i < V; ++i)
            if (logits[i] > logits[best]) best = i;
        return best;
    }
    float maxv = logits[0];
    for (int i = 1; i < V; ++i) maxv = std::max(maxv, logits[i]);
    std::vector<double> probs(V);
    double sum = 0.0;
    for (int i = 0; i < V; ++i) {
        probs[i] = std::exp((logits[i] - maxv) / temperature);
        sum += probs[i];
    }
    std::uniform_real_distribution<double> dist(0.0, sum);
    double r = dist(rng), acc = 0.0;
    for (int i = 0; i < V; ++i) { acc += probs[i]; if (acc >= r) return i; }
    return V - 1;
}

// Helpers for manual ChatML construction
static void push_special(std::vector<uint32_t>& ids, const io::Tokenizer& tok, const std::string& name) {
    ids.push_back(tok.get_id(name));
}

static void push_text(std::vector<uint32_t>& ids, const io::Tokenizer& tok, const std::string& text) {
    auto tmp = tok.encode(text);
    ids.insert(ids.end(), tmp.begin(), tmp.end());
}

int main(int argc, char** argv) {
    if (argc < 2) usage();

    std::string model_dir   = get_flag(argc, argv, "--model");
    std::string prompt      = get_flag(argc, argv, "--prompt");
    std::string expert_path = get_flag(argc, argv, "--expert");
    float expert_scale      = get_flag_f(argc, argv, "--expert-scale", model::kExpertOutputScale);
    int   max_new_tokens    = get_flag_i(argc, argv, "--max-tokens",  128);
    float temperature       = get_flag_f(argc, argv, "--temperature", 0.f);
    int   max_seq           = get_flag_i(argc, argv, "--max-seq",     2048);
    bool  use_chat          = false;

    // Check for --chat flag
    for(int i=1; i<argc; ++i) {
        if(strcmp(argv[i], "--chat") == 0) { use_chat = true; break; }
    }

    if (model_dir.empty() || prompt.empty()) {
        fprintf(stderr, "--model and --prompt are required\n"); usage();
    }

    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));
    CUBLAS_CHECK(cublasSetMathMode(cublas, CUBLAS_TF32_TENSOR_OP_MATH));

    io::Tokenizer tok;
    tok.load(model_dir);

    model::QwenModel mdl;
    model::qwen_load(mdl, model_dir, max_seq);
    printf("Model loaded: %d layers, hidden=%d, vocab=%d\n",
           mdl.cfg.num_layers, mdl.cfg.hidden_size, mdl.cfg.vocab_size);

    model::ExpertModel expert;
    bool has_expert = !expert_path.empty();
    if (has_expert) {
        model::expert_alloc(expert,
                            mdl.cfg.num_layers,
                            mdl.cfg.hidden_size,
                            mdl.cfg.intermediate_size,
                            max_seq);
        io::checkpoint_load(expert_path, expert);
        expert.output_scale = expert_scale;
        printf("Expert loaded from %s  (scale=%.4f)\n", expert_path.c_str(), expert_scale);
    } else {
        printf("No expert — running base model only.\n");
    }

    // Build the prompt tokens
    std::vector<uint32_t> prompt_ids;

    if (use_chat) {
        // ChatML Template:
        // <|im_start|>system\nYou are a helpful coding assistant.<|im_end|>\n
        // <|im_start|>user\n{PROMPT}<|im_end|>\n
        // <|im_start|>assistant\n
        
        push_special(prompt_ids, tok, "<|im_start|>");
        push_text(prompt_ids, tok, "system\nYou are a helpful coding assistant.");
        push_special(prompt_ids, tok, "<|im_end|>");
        push_text(prompt_ids, tok, "\n");

        push_special(prompt_ids, tok, "<|im_start|>");
        push_text(prompt_ids, tok, "user\n" + prompt);
        push_special(prompt_ids, tok, "<|im_end|>");
        push_text(prompt_ids, tok, "\n");

        push_special(prompt_ids, tok, "<|im_start|>");
        push_text(prompt_ids, tok, "assistant\n");
    } else {
        // Raw Mode
        prompt_ids = tok.encode(prompt);
    }

    if (prompt_ids.empty()) { fprintf(stderr, "Prompt encoded to zero tokens.\n"); return 1; }

    int prompt_len = (int)prompt_ids.size();
    if (prompt_len > max_seq - 1) {
        fprintf(stderr, "Prompt too long (%d tokens, max %d).\n", prompt_len, max_seq - 1);
        return 1;
    }
    printf("\nPrompt (%d tokens): %s\n", prompt_len, prompt.c_str());

    std::vector<int32_t> h_tokens(prompt_len);
    for (int i = 0; i < prompt_len; ++i) h_tokens[i] = (int32_t)prompt_ids[i];

    int32_t* d_tokens = (int32_t*)backend::device_alloc(max_seq * sizeof(int32_t));
    backend::host_to_device(d_tokens, h_tokens.data(), prompt_len * sizeof(int32_t));

    int V = mdl.cfg.vocab_size;
    float* d_logits = (float*)backend::device_alloc((size_t)max_seq * V * sizeof(float));
    std::vector<float> h_logits(V);

    printf("\nGenerating:\n");
    fflush(stdout);

    model::qwen_forward(mdl, cublas, d_tokens, prompt_len,
                        has_expert ? &expert : nullptr, d_logits);

    backend::device_to_host(h_logits.data(),
                             d_logits + (size_t)(prompt_len - 1) * V,
                             V * sizeof(float));

    std::mt19937 rng(42);
    int next_token = sample(h_logits.data(), V, temperature, rng);
    int32_t one;

    for (int step = 0; step < max_new_tokens; ++step) {
        if (next_token == (int)tok.eos_id) break;

        printf("%s", tok.decode({(uint32_t)next_token}).c_str());
        fflush(stdout);

        one = (int32_t)next_token;
        backend::host_to_device(d_tokens, &one, sizeof(int32_t));

        model::qwen_forward(mdl, cublas, d_tokens, 1,
                            has_expert ? &expert : nullptr, d_logits);

        backend::device_to_host(h_logits.data(), d_logits, V * sizeof(float));
        next_token = sample(h_logits.data(), V, temperature, rng);
    }
    printf("\n");

    backend::device_free(d_tokens);
    backend::device_free(d_logits);
    model::qwen_free(mdl);
    if (has_expert) model::expert_free(expert);
    cublasDestroy(cublas);
    return 0;
}