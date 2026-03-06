#pragma once
#include <string>
#include <vector>
#include <unordered_map>

namespace io {

enum class DType { BF16, F32, F16, I32, I64 };

struct TensorInfo {
    DType  dtype;
    std::vector<int64_t> shape;
    size_t data_offset;
    size_t nbytes;
};

struct SafeTensors {
    int    fd        = -1;
    void*  mmap_ptr  = nullptr;
    size_t mmap_size = 0;
    const char* data_ptr = nullptr;
    std::unordered_map<std::string, TensorInfo> tensors;

    void   open(const std::string& path);
    void   close();
    size_t load_to_device(const std::string& name, void* dst) const;
    bool   has(const std::string& name) const;
    const std::vector<int64_t>& shape(const std::string& name) const;
};

struct SafeTensorsDir {
    std::vector<SafeTensors> shards;
    void   open(const std::string& dir);
    void   close();
    size_t load_to_device(const std::string& name, void* dst) const;
    bool   has(const std::string& name) const;
};

} // namespace io