// ============================================================
// FILE: trainer/include/io/checkpoint.h
// ============================================================
#pragma once
#include <string>
#include <cstdint>

namespace model { struct ExpertFFN; }

namespace io {

// Checkpoint binary format:
//   [4 bytes magic = 0x45585054]
//   [4 bytes version = 1]
//   [4 bytes num_layers]
//   [4 bytes hidden_size]
//   [4 bytes intermediate_size]
//   For each layer l in 0..num_layers-1:
//     gate_proj fp32 weights [intermediate * hidden]
//     up_proj   fp32 weights [intermediate * hidden]
//     down_proj fp32 weights [hidden * intermediate]
//   (all stored as raw little-endian fp32)
constexpr uint32_t CKPT_MAGIC   = 0x45585054;
constexpr uint32_t CKPT_VERSION = 1;

void checkpoint_save(const std::string& path, const model::ExpertFFN& e);
void checkpoint_load(const std::string& path, model::ExpertFFN& e);

} // namespace io