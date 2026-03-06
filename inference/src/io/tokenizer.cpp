// ============================================================
// FILE: trainer/src/io/tokenizer.cpp
// ============================================================
#include "io/tokenizer.h"
#include <fstream>
#include <nlohmann/json.hpp>
#include <vector>
#include <string>
#include <cassert>
#include <cstdio>

using json = nlohmann::json;

namespace io {

void Tokenizer::load(const std::string& model_dir) {
    std::ifstream f(model_dir + "/tokenizer.json");
    if (!f) { fprintf(stderr, "Cannot open tokenizer.json\n"); exit(1); }
    json j; f >> j;

    // Qwen tokenizer.json structure:
    // { "model": { "vocab": {"token": id, ...}, "merges": ["t1 t2", ...] } }
    const auto& model = j["model"];
    const auto& vocab = model["vocab"];

    for (auto& [tok, id] : vocab.items()) {
        uint32_t tid = id.get<uint32_t>();
        token_to_id[tok] = tid;
        id_to_token[tid] = tok;
    }

    const auto& merges_list = model["merges"];
    merges.reserve(merges_list.size());
    merge_map.reserve(merges_list.size());
    for (size_t rank = 0; rank < merges_list.size(); ++rank) {
        std::string pair = merges_list[rank].get<std::string>();
        // Format: "token_a token_b"
        size_t sp = pair.find(' ');
        if (sp == std::string::npos) continue;
        std::string a = pair.substr(0, sp);
        std::string b = pair.substr(sp+1);
        auto ia = token_to_id.find(a);
        auto ib = token_to_id.find(b);
        if (ia == token_to_id.end() || ib == token_to_id.end()) continue;
        uint32_t left = ia->second, right = ib->second;
        merges.push_back({left, right});
        merge_map[(uint64_t)left << 32 | right] = (uint32_t)rank;
    }

    // Override special tokens if present
    if (j.contains("added_tokens")) {
        for (auto& at : j["added_tokens"]) {
            std::string content = at["content"].get<std::string>();
            uint32_t id = at["id"].get<uint32_t>();
            token_to_id[content] = id;
            id_to_token[id] = content;
        }
    }

    // Read eos/bos from generation_config.json
    std::ifstream gc(model_dir + "/generation_config.json");
    if (gc) {
        json gj; gc >> gj;
        
        // --- FIX: Handle Array vs Number for EOS ---
        if (gj.contains("eos_token_id")) {
            if (gj["eos_token_id"].is_array() && !gj["eos_token_id"].empty()) {
                eos_id = gj["eos_token_id"][0].get<uint32_t>();
            } else if (gj["eos_token_id"].is_number()) {
                eos_id = gj["eos_token_id"].get<uint32_t>();
            }
        }

        // --- FIX: Handle Array vs Number for BOS ---
        if (gj.contains("bos_token_id")) {
            if (gj["bos_token_id"].is_array() && !gj["bos_token_id"].empty()) {
                bos_id = gj["bos_token_id"][0].get<uint32_t>();
            } else if (gj["bos_token_id"].is_number()) {
                bos_id = gj["bos_token_id"].get<uint32_t>();
            }
        }
    }
    pad_id = bos_id;
}

std::vector<uint32_t> Tokenizer::encode(const std::string& text) const {
    if (text.empty()) return {};

    std::vector<uint32_t> ids;
    ids.reserve(text.size());

    for (unsigned char c : text) {
        std::string tok;
        if (c >= 33 && c <= 126) {
            tok = std::string(1, (char)c);
        } else {
            char buf[8]; snprintf(buf, sizeof(buf), "\xc4\xa0");
            tok = std::string(1, (char)c);
        }
        auto it = token_to_id.find(tok);
        if (it != token_to_id.end()) {
            ids.push_back(it->second);
        } else {
            char hex[8]; snprintf(hex, sizeof(hex), "<0x%02X>", c);
            auto it2 = token_to_id.find(hex);
            if (it2 != token_to_id.end()) ids.push_back(it2->second);
        }
    }

    bool changed = true;
    while (changed && ids.size() > 1) {
        changed = false;
        uint32_t best_rank = UINT32_MAX;
        size_t best_pos = 0;
        for (size_t i = 0; i + 1 < ids.size(); ++i) {
            uint64_t key = (uint64_t)ids[i] << 32 | ids[i+1];
            auto it = merge_map.find(key);
            if (it != merge_map.end() && it->second < best_rank) {
                best_rank = it->second;
                best_pos  = i;
            }
        }
        if (best_rank != UINT32_MAX) {
            auto& [left, right] = merges[best_rank];
            std::string merged = id_to_token.at(left) + id_to_token.at(right);
            auto it = token_to_id.find(merged);
            if (it != token_to_id.end()) {
                ids[best_pos] = it->second;
                ids.erase(ids.begin() + best_pos + 1);
                changed = true;
            }
        }
    }
    return ids;
}

std::string Tokenizer::decode(const std::vector<uint32_t>& ids) const {
    std::string result;
    for (uint32_t id : ids) {
        auto it = id_to_token.find(id);
        if (it != id_to_token.end()) result += it->second;
    }
    std::string out;
    for (size_t i = 0; i < result.size(); ) {
        if ((unsigned char)result[i] == 0xC4 && i+1 < result.size() &&
            (unsigned char)result[i+1] == 0xA0) {
            out += ' '; i += 2;
        } else {
            out += result[i++];
        }
    }
    return out;
}

// Ensure this matches the header declaration we fixed earlier
uint32_t Tokenizer::get_id(const std::string& token) const {
    auto it = token_to_id.find(token);
    if (it != token_to_id.end()) {
        return it->second;
    }
    fprintf(stderr, "Warning: special token '%s' not found in vocab.\n", token.c_str());
    return 0;
}

} // namespace io