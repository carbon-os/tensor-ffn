// ============================================================
// FILE: trainer/include/io/tokenizer.h
// ============================================================
#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

namespace io {

// Qwen 2.5 uses tiktoken BPE.
// tokenizer.json has "model": { "vocab": {...}, "merges": [...] }
struct Tokenizer {
    // token_id → utf8 bytes
    std::unordered_map<uint32_t, std::string> id_to_token;
    // utf8 bytes → token_id
    std::unordered_map<std::string, uint32_t> token_to_id;
    // BPE merge rules: (left_id, right_id) → merged_id
    // stored as rank-ordered list (lower index = higher priority)
    std::vector<std::pair<uint32_t, uint32_t>> merges;
    std::unordered_map<uint64_t, uint32_t>     merge_map;  // key = left<<32|right

    uint32_t eos_id = 151645;
    uint32_t bos_id = 151643;
    uint32_t pad_id = 151643;

    void load(const std::string& model_dir);

    // Encode plain text to token ids. BOS is NOT prepended automatically.
    std::vector<uint32_t> encode(const std::string& text) const;

    // Decode token ids to text.
    std::string decode(const std::vector<uint32_t>& ids) const;
};

} // namespace io