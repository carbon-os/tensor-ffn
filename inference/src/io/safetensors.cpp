// ============================================================
// FILE: trainer/src/io/safetensors.cpp
// ============================================================
#include "io/safetensors.h"
#include "backend/memory.h"
#include <nlohmann/json.hpp>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <dirent.h>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

using json = nlohmann::json;

namespace io {

static DType parse_dtype(const std::string& s) {
    if (s == "BF16") return DType::BF16;
    if (s == "F32")  return DType::F32;
    if (s == "F16")  return DType::F16;
    if (s == "I32")  return DType::I32;
    if (s == "I64")  return DType::I64;
    fprintf(stderr, "Unknown dtype: %s\n", s.c_str()); exit(1);
}

static size_t dtype_bytes(DType d) {
    switch (d) {
        case DType::BF16: return 2;
        case DType::F16:  return 2;
        case DType::F32:  return 4;
        case DType::I32:  return 4;
        case DType::I64:  return 8;
    }
    return 0;
}

void SafeTensors::open(const std::string& path) {
    fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) { perror(("Cannot open " + path).c_str()); exit(1); }
    struct stat st; fstat(fd, &st);
    mmap_size = st.st_size;
    mmap_ptr  = mmap(nullptr, mmap_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mmap_ptr == MAP_FAILED) { perror("mmap failed"); exit(1); }

    const char* base = (const char*)mmap_ptr;

    // First 8 bytes: little-endian uint64 header length
    uint64_t header_len;
    memcpy(&header_len, base, 8);

    // Parse header JSON
    std::string header_str(base + 8, header_len);
    json header = json::parse(header_str);

    for (auto& [name, val] : header.items()) {
        if (name == "__metadata__") continue;

        TensorInfo ti;
        ti.dtype = parse_dtype(val["dtype"].get<std::string>());

        for (auto& dim : val["shape"])
            ti.shape.push_back(dim.get<int64_t>());

        auto& offsets  = val["data_offsets"];
        size_t start   = offsets[0].get<size_t>();
        size_t end     = offsets[1].get<size_t>();
        ti.data_offset = start;
        ti.nbytes      = end - start;

        tensors[name] = ti;
    }

    data_ptr = base + 8 + header_len;
}

void SafeTensors::close() {
    if (mmap_ptr && mmap_ptr != MAP_FAILED) {
        munmap(mmap_ptr, mmap_size);
        mmap_ptr = nullptr;
    }
    if (fd >= 0) { ::close(fd); fd = -1; }
}

size_t SafeTensors::load_to_device(const std::string& name, void* dst) const {
    auto it = tensors.find(name);
    if (it == tensors.end()) return 0;
    const auto& ti = it->second;
    backend::host_to_device(dst, data_ptr + ti.data_offset, ti.nbytes);
    return ti.nbytes;
}

bool SafeTensors::has(const std::string& name) const {
    return tensors.count(name) > 0;
}

const std::vector<int64_t>& SafeTensors::shape(const std::string& name) const {
    return tensors.at(name).shape;
}

void SafeTensorsDir::open(const std::string& dir) {
    DIR* d = opendir(dir.c_str());
    if (!d) { perror(("Cannot open dir " + dir).c_str()); exit(1); }
    struct dirent* ent;
    std::vector<std::string> files;
    while ((ent = readdir(d)) != nullptr) {
        std::string name = ent->d_name;
        if (name.size() > 12 && name.substr(name.size()-12) == ".safetensors")
            files.push_back(dir + "/" + name);
    }
    closedir(d);
    std::sort(files.begin(), files.end());
    shards.resize(files.size());
    for (size_t i = 0; i < files.size(); ++i) shards[i].open(files[i]);
}

void SafeTensorsDir::close() {
    for (auto& s : shards) s.close();
}

size_t SafeTensorsDir::load_to_device(const std::string& name, void* dst) const {
    for (auto& s : shards) {
        size_t r = s.load_to_device(name, dst);
        if (r > 0) return r;
    }
    return 0;
}

bool SafeTensorsDir::has(const std::string& name) const {
    for (auto& s : shards) if (s.has(name)) return true;
    return false;
}

} // namespace io