#include "io/checkpoint.h"
#include "backend/memory.h"
#include "backend/cuda_ops.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace io {

void checkpoint_load(const std::string& path, model::ExpertModel& e) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { perror(("Cannot open " + path).c_str()); exit(1); }

    uint32_t magic, version, nl, H, I;
    fread(&magic,   4, 1, f);
    fread(&version, 4, 1, f);
    fread(&nl,      4, 1, f);
    fread(&H,       4, 1, f);
    fread(&I,       4, 1, f);

    if (magic != CKPT_MAGIC) {
        fprintf(stderr, "Bad checkpoint magic: 0x%08x (expected 0x%08x)\n",
                magic, CKPT_MAGIC);
        exit(1);
    }
    if ((int)nl != e.num_layers || (int)H != e.hidden_size || (int)I != e.intermediate_size) {
        fprintf(stderr, "Expert shape mismatch: checkpoint=%d×%d×%d, model=%d×%d×%d\n",
                nl, H, I, e.num_layers, e.hidden_size, e.intermediate_size);
        exit(1);
    }

    size_t gate_up = (size_t)I * H;
    size_t down    = (size_t)H * I;
    std::vector<float> buf;

    for (int l = 0; l < e.num_layers; ++l) {
        auto& el = e.layers[l];

        // gate_proj: fp32 on disk → load to tmp, cast to bf16
        buf.resize(gate_up);

        fread(buf.data(), sizeof(float), gate_up, f);
        // temporary fp32 device buffer for cast
        float* tmp = (float*)backend::device_alloc(gate_up * sizeof(float));
        backend::host_to_device(tmp, buf.data(), gate_up * sizeof(float));
        backend::cast_fp32_to_bf16(tmp, el.gate_proj, (int)gate_up);
        backend::device_free(tmp);

        fread(buf.data(), sizeof(float), gate_up, f);
        tmp = (float*)backend::device_alloc(gate_up * sizeof(float));
        backend::host_to_device(tmp, buf.data(), gate_up * sizeof(float));
        backend::cast_fp32_to_bf16(tmp, el.up_proj, (int)gate_up);
        backend::device_free(tmp);

        buf.resize(down);
        fread(buf.data(), sizeof(float), down, f);
        tmp = (float*)backend::device_alloc(down * sizeof(float));
        backend::host_to_device(tmp, buf.data(), down * sizeof(float));
        backend::cast_fp32_to_bf16(tmp, el.down_proj, (int)down);
        backend::device_free(tmp);
    }

    fclose(f);
}

} // namespace io