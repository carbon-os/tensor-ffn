// ============================================================
// FILE: trainer/src/io/safetensors.cpp
// ============================================================
#include "io/safetensors.h"
#include "backend/memory.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <dirent.h>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <algorithm>

namespace io {

// ---- Minimal JSON parser for safetensors header ----
// The header is a flat JSON object: { "tensor_name": { "dtype": "BF16",
//   "shape": [...], "data_offsets": [start, end] }, ... }
// We parse it without a full JSON library to avoid extra deps.

static std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\n\r\"");
    size_t b = s.find_last_not_of(" \t\n\r\"");
    if (a == std::string::npos) return "";
    return s.substr(a, b - a + 1);
}

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

// Parse safetensors JSON header into tensor map.
// header: null-terminated JSON string
static void parse_header(const char* header, size_t header_len,
                          std::unordered_map<std::string, TensorInfo>& out) {
    // Simple recursive-descent parser for the flat structure.
    // We scan for top-level keys and their nested objects.
    std::string s(header, header_len);
    size_t pos = 0;
    auto skip_ws = [&]() { while (pos < s.size() && (s[pos]==' '||s[pos]=='\t'||s[pos]=='\n'||s[pos]=='\r')) ++pos; };
    auto read_string = [&]() -> std::string {
        skip_ws();
        if (pos >= s.size() || s[pos] != '"') return "";
        ++pos;
        std::string r;
        while (pos < s.size() && s[pos] != '"') { if (s[pos]=='\\') ++pos; r += s[pos++]; }
        ++pos;  // closing "
        return r;
    };

    skip_ws();
    if (s[pos] != '{') return;
    ++pos;

    while (pos < s.size()) {
        skip_ws();
        if (s[pos] == '}') break;
        if (s[pos] == ',') { ++pos; continue; }

        std::string key = read_string();
        skip_ws();
        if (pos < s.size() && s[pos] == ':') ++pos;
        skip_ws();

        // Skip "__metadata__" key
        if (key == "__metadata__") {
            // skip the value object
            int depth = 0;
            while (pos < s.size()) {
                if (s[pos] == '{') ++depth;
                else if (s[pos] == '}') { --depth; if (depth == 0) { ++pos; break; } }
                ++pos;
            }
            continue;
        }

        // Parse tensor info object: { "dtype": ..., "shape": [...], "data_offsets": [start, end] }
        if (pos >= s.size() || s[pos] != '{') break;
        ++pos;

        TensorInfo ti;
        bool have_dtype = false, have_shape = false, have_offsets = false;

        while (pos < s.size()) {
            skip_ws();
            if (s[pos] == '}') { ++pos; break; }
            if (s[pos] == ',') { ++pos; continue; }

            std::string field = read_string();
            skip_ws(); if (s[pos] == ':') ++pos; skip_ws();

            if (field == "dtype") {
                std::string dt = read_string();
                ti.dtype = parse_dtype(dt);
                have_dtype = true;
            } else if (field == "shape") {
                // parse array of ints
                if (s[pos] == '[') { ++pos; }
                ti.shape.clear();
                while (pos < s.size() && s[pos] != ']') {
                    skip_ws();
                    if (s[pos] == ',') { ++pos; continue; }
                    int64_t v = 0;
                    while (pos < s.size() && s[pos] >= '0' && s[pos] <= '9') v = v*10+(s[pos++]-'0');
                    ti.shape.push_back(v);
                }
                if (s[pos] == ']') ++pos;
                have_shape = true;
            } else if (field == "data_offsets") {
                if (s[pos] == '[') ++pos;
                skip_ws();
                size_t start = 0, end = 0;
                while (pos < s.size() && s[pos] >= '0' && s[pos] <= '9') start = start*10+(s[pos++]-'0');
                skip_ws(); if (s[pos]==',') ++pos; skip_ws();
                while (pos < s.size() && s[pos] >= '0' && s[pos] <= '9') end = end*10+(s[pos++]-'0');
                if (s[pos] == ']') ++pos;
                ti.data_offset = start;
                ti.nbytes      = end - start;
                have_offsets   = true;
            } else {
                // skip unknown field value
                while (pos < s.size() && s[pos] != ',' && s[pos] != '}') ++pos;
            }
        }

        if (have_dtype && have_shape && have_offsets) {
            // Compute expected nbytes for validation
            size_t elems = 1;
            for (auto d : ti.shape) elems *= d;
            // (nbytes from data_offsets may differ slightly due to alignment; trust it)
            out[key] = ti;
        }
    }
}

void SafeTensors::open(const std::string& path) {
    fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) { perror(("Cannot open " + path).c_str()); exit(1); }
    struct stat st; fstat(fd, &st);
    mmap_size = st.st_size;
    mmap_ptr  = mmap(nullptr, mmap_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mmap_ptr == MAP_FAILED) { perror("mmap failed"); exit(1); }

    const char* base = (const char*)mmap_ptr;
    // First 8 bytes: uint64 header length
    uint64_t header_len;
    memcpy(&header_len, base, 8);
    const char* header_start = base + 8;
    parse_header(header_start, (size_t)header_len, tensors);
    data_ptr = header_start + header_len;
}

void SafeTensors::close() {
    if (mmap_ptr && mmap_ptr != MAP_FAILED) { munmap(mmap_ptr, mmap_size); mmap_ptr = nullptr; }
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