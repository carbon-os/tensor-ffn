#pragma once
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t s = (call); \
    if (s != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)s, __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)

namespace backend {

// GEMM: C(fp32) = alpha * op(A[bf16]) * op(B[bf16]) + beta * C
void gemm_bf16(cublasHandle_t, int M, int N, int K,
               float alpha,
               const __nv_bfloat16* A, const __nv_bfloat16* B,
               float beta, float* C,
               bool transA, bool transB);

// GEMM: C(bf16) = op(A[bf16]) * op(B[bf16])
void gemm_bf16_out(cublasHandle_t, int M, int N, int K,
                   const __nv_bfloat16* A, const __nv_bfloat16* B,
                   __nv_bfloat16* C, bool transA, bool transB);

// GEMM: C(fp32) = alpha * op(A[fp32]) * op(B[fp32]) + beta * C
void gemm_fp32(cublasHandle_t, int M, int N, int K,
               float alpha,
               const float* A, const float* B,
               float beta, float* C,
               bool transA, bool transB);

// Fused SiLU-mul: out[i] = SiLU(gate[i]) * up[i]  (bf16)
void silu_mul_bf16(const __nv_bfloat16* gate, const __nv_bfloat16* up,
                   __nv_bfloat16* out, int n, cudaStream_t s = 0);

// In-place row-wise softmax (fp32)
void softmax_fp32(float* x, int rows, int cols, cudaStream_t s = 0);

// RMSNorm forward — rms_scratch[row] = rsqrt(mean(x^2)+eps)
void rmsnorm_bf16(const __nv_bfloat16* x, const __nv_bfloat16* w,
                  __nv_bfloat16* out, float* rms_scratch,
                  int rows, int dim, float eps, cudaStream_t s = 0);

// Cast helpers
void cast_bf16_to_fp32(const __nv_bfloat16* src, float* dst, int n, cudaStream_t s = 0);
void cast_fp32_to_bf16(const float* src, __nv_bfloat16* dst, int n, cudaStream_t s = 0);

// Element-wise add (fp32)
void add_fp32(const float* a, const float* b, float* c, int n, cudaStream_t s = 0);

// In-place scale (fp32)
void scale_fp32(float* x, float scale, int n, cudaStream_t s = 0);

} // namespace backend