// ============================================================
// FILE: trainer/include/backend/memory.h
// ============================================================
#pragma once
#include <cstddef>
#include <cuda_runtime.h>

namespace backend {

// --------------- error helpers ---------------
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t _e = (call);                                            \
        if (_e != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(_e));            \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

#define CUBLAS_CHECK(call)                                                  \
    do {                                                                    \
        cublasStatus_t _s = (call);                                         \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                  \
            fprintf(stderr, "cuBLAS error %s:%d: %d\n",                    \
                    __FILE__, __LINE__, (int)_s);                           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// --------------- raw device helpers ---------------
void* device_alloc(size_t bytes);
void  device_free(void* ptr);
void  host_to_device(void* dst, const void* src, size_t bytes);
void  device_to_host(void* dst, const void* src, size_t bytes);
void  device_memset(void* ptr, int val, size_t bytes);
void  device_copy(void* dst, const void* src, size_t bytes);   // device→device

// --------------- RAII wrapper ---------------
struct DeviceBuf {
    void*  ptr  = nullptr;
    size_t size = 0;

    DeviceBuf() = default;
    explicit DeviceBuf(size_t bytes);
    ~DeviceBuf();

    DeviceBuf(const DeviceBuf&)            = delete;
    DeviceBuf& operator=(const DeviceBuf&) = delete;
    DeviceBuf(DeviceBuf&&) noexcept;
    DeviceBuf& operator=(DeviceBuf&&) noexcept;

    void alloc(size_t bytes);
    void zero();

    template<typename T>       T* as()       { return static_cast<T*>(ptr); }
    template<typename T> const T* as() const { return static_cast<const T*>(ptr); }
};

} // namespace backend