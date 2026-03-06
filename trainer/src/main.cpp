// ============================================================
// FILE: trainer/src/main.cpp
// ============================================================
#include "training/trainer.h"
#include "io/corpus.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static void usage() {
    fprintf(stderr,
        "tensor-ffn trainer\n"
        "\n"
        "Subcommands:\n"
        "  tokenize  --model <dir> --in <path> --out <corpus.bin>\n"
        "  train     --model <dir> --data <corpus.bin> --out <expert.bin>\n"
        "            [--steps N] [--lr F] [--seq N]\n"
        "\n");
    exit(1);
}

static std::string get_flag(int argc, char** argv, const char* flag, const char* def = "") {
    for (int i = 1; i < argc - 1; ++i)
        if (strcmp(argv[i], flag) == 0) return argv[i+1];
    return def;
}
static int get_flag_i(int argc, char** argv, const char* flag, int def = 0) {
    std::string v = get_flag(argc, argv, flag, "");
    return v.empty() ? def : std::stoi(v);
}
static float get_flag_f(int argc, char** argv, const char* flag, float def = 0.0f) {
    std::string v = get_flag(argc, argv, flag, "");
    return v.empty() ? def : std::stof(v);
}

int main(int argc, char** argv) {
    if (argc < 2) usage();
    std::string cmd = argv[1];

    if (cmd == "tokenize") {
        std::string model = get_flag(argc, argv, "--model");
        std::string in    = get_flag(argc, argv, "--in");
        std::string out   = get_flag(argc, argv, "--out");
        if (model.empty() || in.empty() || out.empty()) {
            fprintf(stderr, "tokenize requires --model --in --out\n"); usage();
        }
        io::tokenize_dir(model, in, out);

    } else if (cmd == "train") {
        training::TrainConfig cfg;
        cfg.model_dir   = get_flag(argc, argv, "--model");
        cfg.corpus_path = get_flag(argc, argv, "--data");
        cfg.out_path    = get_flag(argc, argv, "--out");
        if (cfg.model_dir.empty() || cfg.corpus_path.empty() || cfg.out_path.empty()) {
            fprintf(stderr, "train requires --model --data --out\n"); usage();
        }
        cfg.steps       = get_flag_i(argc, argv, "--steps", 0);
        cfg.lr          = get_flag_f(argc, argv, "--lr", 3e-4f);
        cfg.seq_len     = get_flag_i(argc, argv, "--seq", 2048);
        cfg.batch_size  = get_flag_i(argc, argv, "--batch", 0);
        cfg.log_every   = get_flag_i(argc, argv, "--log-every", 10);
        cfg.save_every  = get_flag_i(argc, argv, "--save-every", 1000);

        training::TrainContext ctx;
        training::trainer_init(ctx, cfg);
        training::trainer_run(ctx, cfg);
        training::trainer_free(ctx);

    } else {
        fprintf(stderr, "Unknown subcommand: %s\n", cmd.c_str());
        usage();
    }
    return 0;
}