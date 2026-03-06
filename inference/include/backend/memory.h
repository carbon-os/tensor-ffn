#pragma once
#include <cstddef>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d — %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } \
} while (0)

namespace backend {

void* device_alloc(size_t bytes);
void  device_free(void* ptr);
void  host_to_device(void* dst, const void* src, size_t bytes);
void  device_to_host(void* dst, const void* src, size_t bytes);
void  device_memset(void* ptr, int val, size_t bytes);
void  device_copy(void* dst, const void* src, size_t bytes);

struct DeviceBuf {
    void*  ptr  = nullptr;
    size_t size = 0;
    DeviceBuf() = default;
    explicit DeviceBuf(size_t bytes);
    ~DeviceBuf();
    DeviceBuf(DeviceBuf&&) noexcept;
    DeviceBuf& operator=(DeviceBuf&&) noexcept;
    void alloc(size_t bytes);
    void zero();
    template<typename T> T* as() { return reinterpret_cast<T*>(ptr); }
};

} // namespace backend