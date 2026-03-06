// ============================================================
// FILE: trainer/include/io/safetensors.h
// ============================================================
#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include <cstdint>

namespace io {

enum class DType { BF16, F32, F16, I32, I64 };

struct TensorInfo {
    DType              dtype;
    std::vector<int64_t> shape;
    size_t             data_offset;  // byte offset into raw data region
    size_t             nbytes;
};

// Memory-mapped safetensors file.
struct SafeTensors {
    int         fd        = -1;
    void*       mmap_ptr  = nullptr;
    size_t      mmap_size = 0;
    const char* data_ptr  = nullptr;  // points into mmap_ptr at data region start

    std::unordered_map<std::string, TensorInfo> tensors;

    // Open file and parse header. Does NOT load weights into GPU.
    void open(const std::string& path);
    void close();

    // Copy named tensor to pre-allocated device buffer.
    // Returns bytes copied, or 0 if not found.
    size_t load_to_device(const std::string& name, void* dst) const;

    // Returns true if tensor exists.
    bool has(const std::string& name) const;

    // Returns shape of named tensor.
    const std::vector<int64_t>& shape(const std::string& name) const;

    ~SafeTensors() { close(); }
};

// Scan directory for all *.safetensors files and aggregate their tensor maps.
// Used to load a model split across multiple shards.
struct SafeTensorsDir {
    std::vector<SafeTensors> shards;

    void open(const std::string& dir);
    void close();

    size_t load_to_device(const std::string& name, void* dst) const;
    bool   has(const std::string& name) const;
};

} // namespace io