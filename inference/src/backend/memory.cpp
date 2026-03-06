#include "backend/memory.h"
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

namespace backend {

void* device_alloc(size_t bytes) {
    void* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, bytes));
    return ptr;
}

void device_free(void* ptr)                          { if (ptr) CUDA_CHECK(cudaFree(ptr)); }
void host_to_device(void* d, const void* h, size_t n){ CUDA_CHECK(cudaMemcpy(d, h, n, cudaMemcpyHostToDevice)); }
void device_to_host(void* h, const void* d, size_t n){ CUDA_CHECK(cudaMemcpy(h, d, n, cudaMemcpyDeviceToHost)); }
void device_memset(void* p, int v, size_t n)          { CUDA_CHECK(cudaMemset(p, v, n)); }
void device_copy(void* d, const void* s, size_t n)    { CUDA_CHECK(cudaMemcpy(d, s, n, cudaMemcpyDeviceToDevice)); }

DeviceBuf::DeviceBuf(size_t bytes) { alloc(bytes); }
DeviceBuf::~DeviceBuf() { if (ptr) { cudaFree(ptr); ptr = nullptr; size = 0; } }
DeviceBuf::DeviceBuf(DeviceBuf&& o) noexcept : ptr(o.ptr), size(o.size) { o.ptr=nullptr; o.size=0; }
DeviceBuf& DeviceBuf::operator=(DeviceBuf&& o) noexcept {
    if (this != &o) { if (ptr) cudaFree(ptr); ptr=o.ptr; size=o.size; o.ptr=nullptr; o.size=0; }
    return *this;
}
void DeviceBuf::alloc(size_t bytes) {
    if (ptr) { cudaFree(ptr); ptr = nullptr; }
    size = bytes;
    CUDA_CHECK(cudaMalloc(&ptr, bytes));
}
void DeviceBuf::zero() { if (ptr && size) CUDA_CHECK(cudaMemset(ptr, 0, size)); }

} // namespace backend