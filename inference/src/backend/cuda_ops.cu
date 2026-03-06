#include "backend/cuda_ops.h"
#include "backend/memory.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#define BLOCK 256

namespace backend {

// ================================================================
//  GEMM wrappers
// ================================================================

void gemm_bf16(cublasHandle_t h, int M, int N, int K,
               float alpha, const __nv_bfloat16* A, const __nv_bfloat16* B,
               float beta, float* C, bool tA, bool tB) {
    auto opA = tA ? CUBLAS_OP_T : CUBLAS_OP_N;
    auto opB = tB ? CUBLAS_OP_T : CUBLAS_OP_N;
    int lda = tA ? M : K, ldb = tB ? K : N;
    CUBLAS_CHECK(cublasGemmEx(h,
        opB, opA, N, M, K, &alpha,
        B, CUDA_R_16BF, ldb, A, CUDA_R_16BF, lda, &beta,
        C, CUDA_R_32F, N,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

void gemm_bf16_out(cublasHandle_t h, int M, int N, int K,
                   const __nv_bfloat16* A, const __nv_bfloat16* B,
                   __nv_bfloat16* C, bool tA, bool tB) {
    auto opA = tA ? CUBLAS_OP_T : CUBLAS_OP_N;
    auto opB = tB ? CUBLAS_OP_T : CUBLAS_OP_N;
    int lda = tA ? M : K, ldb = tB ? K : N;
    float alpha = 1.f, beta = 0.f;
    CUBLAS_CHECK(cublasGemmEx(h,
        opB, opA, N, M, K, &alpha,
        B, CUDA_R_16BF, ldb, A, CUDA_R_16BF, lda, &beta,
        C, CUDA_R_16BF, N,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

void gemm_fp32(cublasHandle_t h, int M, int N, int K,
               float alpha, const float* A, const float* B,
               float beta, float* C, bool tA, bool tB) {
    auto opA = tA ? CUBLAS_OP_T : CUBLAS_OP_N;
    auto opB = tB ? CUBLAS_OP_T : CUBLAS_OP_N;
    int lda = tA ? M : K, ldb = tB ? K : N;
    CUBLAS_CHECK(cublasSgemm(h, opB, opA, N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

// ================================================================
//  SiLU-mul
// ================================================================

__global__ static void k_silu_mul_bf16(const __nv_bfloat16* gate, const __nv_bfloat16* up,
                                        __nv_bfloat16* out, int n) {
    int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i >= n) return;
    float g = __bfloat162float(gate[i]);
    float u = __bfloat162float(up[i]);
    out[i]  = __float2bfloat16((g / (1.f + expf(-g))) * u);
}
void silu_mul_bf16(const __nv_bfloat16* gate, const __nv_bfloat16* up,
                   __nv_bfloat16* out, int n, cudaStream_t s) {
    k_silu_mul_bf16<<<(n+BLOCK-1)/BLOCK, BLOCK, 0, s>>>(gate, up, out, n);
}

// ================================================================
//  RMSNorm forward
// ================================================================

__global__ static void k_rmsnorm(const __nv_bfloat16* x, const __nv_bfloat16* w,
                                  __nv_bfloat16* out, float* rms_out,
                                  int rows, int dim, float eps) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const __nv_bfloat16* xr = x   + row * dim;
    __nv_bfloat16*       or_ = out + row * dim;

    float ss = 0.f;
    for (int i = threadIdx.x; i < dim; i += BLOCK) { float v = __bfloat162float(xr[i]); ss += v*v; }
    __shared__ float sh[BLOCK];
    sh[threadIdx.x] = ss; __syncthreads();
    for (int s = BLOCK>>1; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x+s];
        __syncthreads();
    }
    float r = rsqrtf(sh[0] / dim + eps);
    if (threadIdx.x == 0 && rms_out) rms_out[row] = r;
    __syncthreads();
    for (int i = threadIdx.x; i < dim; i += BLOCK)
        or_[i] = __float2bfloat16(__bfloat162float(xr[i]) * r * __bfloat162float(w[i]));
}
void rmsnorm_bf16(const __nv_bfloat16* x, const __nv_bfloat16* w,
                  __nv_bfloat16* out, float* rms, int rows, int dim, float eps, cudaStream_t s) {
    k_rmsnorm<<<rows, BLOCK, 0, s>>>(x, w, out, rms, rows, dim, eps);
}

// ================================================================
//  Softmax
// ================================================================

__global__ static void k_softmax(float* x, int rows, int cols) {
    int row = blockIdx.x; if (row >= rows) return;
    float* xr = x + row * cols;
    float mx = -1e30f;
    for (int i = threadIdx.x; i < cols; i += BLOCK) mx = fmaxf(mx, xr[i]);
    __shared__ float sm[BLOCK]; sm[threadIdx.x] = mx; __syncthreads();
    for (int s = BLOCK>>1; s > 0; s >>= 1) {
        if (threadIdx.x < s) sm[threadIdx.x] = fmaxf(sm[threadIdx.x], sm[threadIdx.x+s]);
        __syncthreads();
    }
    mx = sm[0];
    float sv = 0.f;
    for (int i = threadIdx.x; i < cols; i += BLOCK) { float v = expf(xr[i]-mx); xr[i]=v; sv+=v; }
    __shared__ float ss[BLOCK]; ss[threadIdx.x] = sv; __syncthreads();
    for (int s = BLOCK>>1; s > 0; s >>= 1) {
        if (threadIdx.x < s) ss[threadIdx.x] += ss[threadIdx.x+s];
        __syncthreads();
    }
    sv = ss[0];
    for (int i = threadIdx.x; i < cols; i += BLOCK) xr[i] /= sv;
}
void softmax_fp32(float* x, int rows, int cols, cudaStream_t s) {
    k_softmax<<<rows, BLOCK, 0, s>>>(x, rows, cols);
}

// ================================================================
//  Cast
// ================================================================

__global__ static void k_bf16_to_fp32(const __nv_bfloat16* s, float* d, int n) {
    int i = blockIdx.x*BLOCK+threadIdx.x; if (i<n) d[i]=__bfloat162float(s[i]);
}
__global__ static void k_fp32_to_bf16(const float* s, __nv_bfloat16* d, int n) {
    int i = blockIdx.x*BLOCK+threadIdx.x; if (i<n) d[i]=__float2bfloat16(s[i]);
}
void cast_bf16_to_fp32(const __nv_bfloat16* s, float* d, int n, cudaStream_t st) {
    k_bf16_to_fp32<<<(n+BLOCK-1)/BLOCK,BLOCK,0,st>>>(s,d,n);
}
void cast_fp32_to_bf16(const float* s, __nv_bfloat16* d, int n, cudaStream_t st) {
    k_fp32_to_bf16<<<(n+BLOCK-1)/BLOCK,BLOCK,0,st>>>(s,d,n);
}

// ================================================================
//  Misc
// ================================================================

__global__ static void k_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x*BLOCK+threadIdx.x; if(i<n) c[i]=a[i]+b[i];
}
void add_fp32(const float* a, const float* b, float* c, int n, cudaStream_t s) {
    k_add<<<(n+BLOCK-1)/BLOCK,BLOCK,0,s>>>(a,b,c,n);
}

__global__ static void k_scale(float* x, float sc, int n) {
    int i = blockIdx.x*BLOCK+threadIdx.x; if(i<n) x[i]*=sc;
}
void scale_fp32(float* x, float sc, int n, cudaStream_t s) {
    k_scale<<<(n+BLOCK-1)/BLOCK,BLOCK,0,s>>>(x,sc,n);
}

} // namespace backend