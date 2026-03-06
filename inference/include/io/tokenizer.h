#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

namespace io {

struct Tokenizer {
    std::unordered_map<std::string, uint32_t> token_to_id;
    std::unordered_map<uint32_t, std::string> id_to_token;
    std::vector<std::pair<uint32_t, uint32_t>> merges;
    std::unordered_map<uint64_t, uint32_t>     merge_map;
    uint32_t eos_id = 151645;
    uint32_t bos_id = 151643;
    uint32_t pad_id = 151643;

    void load(const std::string& model_dir);
    std::vector<uint32_t> encode(const std::string& text) const;
    std::string           decode(const std::vector<uint32_t>& ids) const;
    
    // MOVED INSIDE THE STRUCT:
    uint32_t get_id(const std::string& token) const;
};

} // namespace io