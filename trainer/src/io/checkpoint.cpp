// ============================================================
// FILE: trainer/src/io/checkpoint.cpp
// ============================================================
#include "io/checkpoint.h"
#include "model/ffn_expert.h"
#include "backend/memory.h"
#include "backend/cuda_ops.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace io {

void checkpoint_save(const std::string& path, const model::ExpertFFN& e) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) { perror(("Cannot save checkpoint " + path).c_str()); exit(1); }

    uint32_t magic   = CKPT_MAGIC;
    uint32_t version = CKPT_VERSION;
    uint32_t nl = (uint32_t)e.num_layers;
    uint32_t H  = (uint32_t)e.hidden_size;
    uint32_t I  = (uint32_t)e.intermediate_size;

    fwrite(&magic,   4, 1, f);
    fwrite(&version, 4, 1, f);
    fwrite(&nl,      4, 1, f);
    fwrite(&H,       4, 1, f);
    fwrite(&I,       4, 1, f);

    size_t gate_up = (size_t)I * H;
    size_t down    = (size_t)H * I;
    std::vector<float> buf;

    for (int l = 0; l < e.num_layers; ++l) {
        auto& el = e.layers[l];

        buf.resize(gate_up);
        backend::device_to_host(buf.data(), el.gate_proj_fp32, gate_up * sizeof(float));
        fwrite(buf.data(), sizeof(float), gate_up, f);

        backend::device_to_host(buf.data(), el.up_proj_fp32, gate_up * sizeof(float));
        fwrite(buf.data(), sizeof(float), gate_up, f);

        buf.resize(down);
        backend::device_to_host(buf.data(), el.down_proj_fp32, down * sizeof(float));
        fwrite(buf.data(), sizeof(float), down, f);
    }

    fclose(f);
    printf("Checkpoint saved to %s\n", path.c_str());
}

void checkpoint_load(const std::string& path, model::ExpertFFN& e) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { perror(("Cannot load checkpoint " + path).c_str()); exit(1); }

    uint32_t magic, version, nl, H, I;
    fread(&magic,   4, 1, f);
    fread(&version, 4, 1, f);
    fread(&nl,      4, 1, f);
    fread(&H,       4, 1, f);
    fread(&I,       4, 1, f);

    if (magic != CKPT_MAGIC) {
        fprintf(stderr, "Bad checkpoint magic: %08x\n", magic); exit(1);
    }
    if ((int)nl != e.num_layers || (int)H != e.hidden_size || (int)I != e.intermediate_size) {
        fprintf(stderr, "Checkpoint shape mismatch: got %d layers %d hidden %d inter\n", nl, H, I);
        exit(1);
    }

    size_t gate_up = (size_t)I * H;
    size_t down    = (size_t)H * I;
    std::vector<float> buf;

    for (int l = 0; l < e.num_layers; ++l) {
        auto& el = e.layers[l];

        buf.resize(gate_up);
        fread(buf.data(), sizeof(float), gate_up, f);
        backend::host_to_device(el.gate_proj_fp32, buf.data(), gate_up * sizeof(float));
        backend::cast_fp32_to_bf16(el.gate_proj_fp32, el.gate_proj, (int)gate_up);

        fread(buf.data(), sizeof(float), gate_up, f);
        backend::host_to_device(el.up_proj_fp32, buf.data(), gate_up * sizeof(float));
        backend::cast_fp32_to_bf16(el.up_proj_fp32, el.up_proj, (int)gate_up);

        buf.resize(down);
        fread(buf.data(), sizeof(float), down, f);
        backend::host_to_device(el.down_proj_fp32, buf.data(), down * sizeof(float));
        backend::cast_fp32_to_bf16(el.down_proj_fp32, el.down_proj, (int)down);
    }

    fclose(f);
    printf("Checkpoint loaded from %s\n", path.c_str());
}

} // namespace io