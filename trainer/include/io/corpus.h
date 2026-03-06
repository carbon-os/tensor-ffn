// ============================================================
// FILE: trainer/include/io/corpus.h
// ============================================================
#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <fstream>

namespace io {

// Binary corpus file format:
//   [4 bytes magic = 0x54464E4E]
//   [4 bytes version = 1]
//   [8 bytes token_count (uint64)]
//   [token_count × uint32  token ids]
constexpr uint32_t CORPUS_MAGIC   = 0x54464E4E;
constexpr uint32_t CORPUS_VERSION = 1;

// Write a tokenized corpus to disk.
void corpus_write(const std::string& path,
                  const std::vector<uint32_t>& tokens);

// Lightweight loader: mmaps the corpus file and slices it into windows.
struct CorpusLoader {
    int      fd         = -1;
    void*    mmap_ptr   = nullptr;
    size_t   mmap_size  = 0;
    uint64_t num_tokens = 0;
    const uint32_t* token_ptr = nullptr;  // points into mmap

    void open(const std::string& path);
    void close();
    ~CorpusLoader() { close(); }

    uint64_t total_tokens() const { return num_tokens; }

    // Fill input/target buffers with one batch.
    // Windows are non-overlapping and packed; wraps around at end of corpus.
    // input:  [batch, seq]  uint32 on host
    // target: [batch, seq]  uint32 on host  (shifted by 1: target[t] = input[t+1])
    // offset: starting token index into corpus (advances by batch*seq each call)
    // Returns false when corpus is exhausted (offset >= num_tokens - seq).
    bool next_batch(uint32_t*  input,
                    uint32_t*  target,
                    int        batch,
                    int        seq,
                    uint64_t&  offset) const;
};

} // namespace io