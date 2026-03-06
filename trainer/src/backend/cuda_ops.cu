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
//  cuBLAS GEMM wrappers
// ================================================================

void gemm_bf16(cublasHandle_t handle,
               int M, int N, int K,
               float alpha,
               const __nv_bfloat16* A,
               const __nv_bfloat16* B,
               float beta,
               float* C,
               bool transA, bool transB) {
    cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
    int lda = transA ? M : K;
    int ldb = transB ? K : N;
    CUBLAS_CHECK(cublasGemmEx(handle,
        opB, opA,
        N, M, K,
        &alpha,
        B, CUDA_R_16BF, ldb,
        A, CUDA_R_16BF, lda,
        &beta,
        C, CUDA_R_32F, N,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

void gemm_bf16_out(cublasHandle_t handle,
                   int M, int N, int K,
                   const __nv_bfloat16* A,
                   const __nv_bfloat16* B,
                   __nv_bfloat16* C,
                   bool transA, bool transB) {
    cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
    int lda = transA ? M : K;
    int ldb = transB ? K : N;
    float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(handle,
        opB, opA,
        N, M, K,
        &alpha,
        B, CUDA_R_16BF, ldb,
        A, CUDA_R_16BF, lda,
        &beta,
        C, CUDA_R_16BF, N,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

void gemm_fp32(cublasHandle_t handle,
               int M, int N, int K,
               float alpha,
               const float* A,
               const float* B,
               float beta,
               float* C,
               bool transA, bool transB) {
    cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
    int lda = transA ? M : K;
    int ldb = transB ? K : N;
    CUBLAS_CHECK(cublasSgemm(handle,
        opB, opA,
        N, M, K,
        &alpha,
        B, ldb,
        A, lda,
        &beta,
        C, N));
}

// ================================================================
//  Element-wise kernels
// ================================================================

__global__ static void k_silu_bf16(const __nv_bfloat16* __restrict__ x,
                                    __nv_bfloat16* __restrict__ out,
                                    int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = __bfloat162float(x[i]);
    out[i]  = __float2bfloat16(v / (1.0f + expf(-v)));
}

void silu_bf16(const __nv_bfloat16* x, __nv_bfloat16* out, int n, cudaStream_t stream) {
    k_silu_bf16<<<(n+BLOCK-1)/BLOCK, BLOCK, 0, stream>>>(x, out, n);
}

__global__ static void k_silu_mul_bf16(const __nv_bfloat16* __restrict__ gate,
                                        const __nv_bfloat16* __restrict__ up,
                                        __nv_bfloat16* __restrict__ out,
                                        int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = __bfloat162float(gate[i]);
    float u = __bfloat162float(up[i]);
    out[i]  = __float2bfloat16((g / (1.0f + expf(-g))) * u);
}

void silu_mul_bf16(const __nv_bfloat16* gate, const __nv_bfloat16* up,
                   __nv_bfloat16* out, int n, cudaStream_t stream) {
    k_silu_mul_bf16<<<(n+BLOCK-1)/BLOCK, BLOCK, 0, stream>>>(gate, up, out, n);
}

__global__ static void k_silu_mul_backward_fp32(const float* __restrict__ d_out,
                                                  const float* __restrict__ gate,
                                                  const float* __restrict__ up,
                                                  float* __restrict__ d_gate,
                                                  float* __restrict__ d_up,
                                                  int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g   = gate[i];
    float u   = up[i];
    float dо  = d_out[i];
    float sig = 1.0f / (1.0f + expf(-g));
    d_gate[i] = dо * u * sig * (1.0f + g * (1.0f - sig));
    d_up[i]   = dо * (g * sig);
}

void silu_mul_backward_fp32(const float* d_out, const float* gate, const float* up,
                             float* d_gate, float* d_up, int n, cudaStream_t stream) {
    k_silu_mul_backward_fp32<<<(n+BLOCK-1)/BLOCK, BLOCK, 0, stream>>>(
        d_out, gate, up, d_gate, d_up, n);
}

// ================================================================
//  RMSNorm
// ================================================================

__global__ static void k_rmsnorm_fwd(const __nv_bfloat16* __restrict__ x,
                                      const __nv_bfloat16* __restrict__ w,
                                      __nv_bfloat16* __restrict__ out,
                                      float* __restrict__ rms_out,
                                      int rows, int dim, float eps) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const __nv_bfloat16* xr = x   + row * dim;
    __nv_bfloat16*       or_ = out + row * dim;

    float ss = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = __bfloat162float(xr[i]);
        ss += v * v;
    }
    __shared__ float shared[BLOCK];
    shared[threadIdx.x] = ss;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    float rms = rsqrtf(shared[0] / dim + eps);
    if (threadIdx.x == 0) rms_out[row] = rms;
    __syncthreads();
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = __bfloat162float(xr[i]) * rms * __bfloat162float(w[i]);
        or_[i]  = __float2bfloat16(v);
    }
}

void rmsnorm_bf16(const __nv_bfloat16* x, const __nv_bfloat16* w,
                  __nv_bfloat16* out, float* rms,
                  int rows, int dim, float eps, cudaStream_t stream) {
    k_rmsnorm_fwd<<<rows, BLOCK, 0, stream>>>(x, w, out, rms, rows, dim, eps);
}

__global__ static void k_rmsnorm_bwd(const float* __restrict__ d_out,
                                      const __nv_bfloat16* __restrict__ x,
                                      const __nv_bfloat16* __restrict__ w,
                                      const float* __restrict__ rms,
                                      float* __restrict__ d_x,
                                      int rows, int dim) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const float* dor = d_out + row * dim;
    const __nv_bfloat16* xr  = x     + row * dim;
    float* dxr = d_x   + row * dim;
    float r = rms[row];

    // 1. Compute dot product (Sum of d_out * w * x)
    float dot = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float wi = __bfloat162float(w[i]);
        float xi = __bfloat162float(xr[i]);
        dot += dor[i] * wi * xi;
    }
    __shared__ float sdot[BLOCK];
    sdot[threadIdx.x] = dot;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) sdot[threadIdx.x] += sdot[threadIdx.x + stride];
        __syncthreads();
    }
    
    // 2. Pre-calculate the subtraction term scaling
    // sum_term = (Sum * r^2) / N
    float sum_term = sdot[0] * r * r / dim;

    // 3. Compute final gradient
    // d_x = r * (d_out * w - sum_term * x)
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float wi = __bfloat162float(w[i]);
        float xi = __bfloat162float(xr[i]);
        
        // FIXED: Removed the extra (* r * r) that was causing gradient explosion
        dxr[i]   = r * (dor[i] * wi - sum_term * xi);
    }
}

void rmsnorm_backward_fp32(const float* d_out, const __nv_bfloat16* x,
                            const __nv_bfloat16* w, const float* rms,
                            float* d_x, int rows, int dim, cudaStream_t stream) {
    k_rmsnorm_bwd<<<rows, BLOCK, 0, stream>>>(d_out, x, w, rms, d_x, rows, dim);
}

// ================================================================
//  Softmax
// ================================================================

__global__ static void k_softmax_fp32(float* __restrict__ x, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    float* xr = x + row * cols;

    float maxv = -1e30f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) maxv = fmaxf(maxv, xr[i]);
    __shared__ float smax[BLOCK];
    smax[threadIdx.x] = maxv;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) smax[threadIdx.x] = fmaxf(smax[threadIdx.x], smax[threadIdx.x + s]);
        __syncthreads();
    }
    maxv = smax[0];

    float sumv = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = expf(xr[i] - maxv);
        xr[i]   = v;
        sumv    += v;
    }
    __shared__ float ssum[BLOCK];
    ssum[threadIdx.x] = sumv;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) ssum[threadIdx.x] += ssum[threadIdx.x + s];
        __syncthreads();
    }
    sumv = ssum[0];
    for (int i = threadIdx.x; i < cols; i += blockDim.x) xr[i] /= sumv;
}

void softmax_fp32(float* x, int rows, int cols, cudaStream_t stream) {
    k_softmax_fp32<<<rows, BLOCK, 0, stream>>>(x, rows, cols);
}

__global__ static void k_softmax_bwd(const float* __restrict__ p,
                                      const float* __restrict__ d_out,
                                      float* __restrict__ d_x,
                                      int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const float* pr  = p     + row * cols;
    const float* dor = d_out + row * cols;
    float*       dxr = d_x   + row * cols;

    float dot = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) dot += pr[i] * dor[i];
    __shared__ float sdot[BLOCK];
    sdot[threadIdx.x] = dot;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdot[threadIdx.x] += sdot[threadIdx.x + s];
        __syncthreads();
    }
    dot = sdot[0];
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        dxr[i] = pr[i] * (dor[i] - dot);
}

void softmax_backward_fp32(const float* p, const float* d_out,
                            float* d_x, int rows, int cols, cudaStream_t stream) {
    k_softmax_bwd<<<rows, BLOCK, 0, stream>>>(p, d_out, d_x, rows, cols);
}

// ================================================================
//  Cast kernels
// ================================================================

__global__ static void k_bf16_to_fp32(const __nv_bfloat16* __restrict__ src,
                                        float* __restrict__ dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __bfloat162float(src[i]);
}

__global__ static void k_fp32_to_bf16(const float* __restrict__ src,
                                        __nv_bfloat16* __restrict__ dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2bfloat16(src[i]);
}

void cast_bf16_to_fp32(const __nv_bfloat16* src, float* dst, int n, cudaStream_t stream) {
    k_bf16_to_fp32<<<(n+BLOCK-1)/BLOCK, BLOCK, 0, stream>>>(src, dst, n);
}

void cast_fp32_to_bf16(const float* src, __nv_bfloat16* dst, int n, cudaStream_t stream) {
    k_fp32_to_bf16<<<(n+BLOCK-1)/BLOCK, BLOCK, 0, stream>>>(src, dst, n);
}

// ================================================================
//  Misc
// ================================================================

__global__ static void k_add_fp32(const float* __restrict__ a,
                                   const float* __restrict__ b,
                                   float* __restrict__ c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

void add_fp32(const float* a, const float* b, float* c, int n, cudaStream_t stream) {
    k_add_fp32<<<(n+BLOCK-1)/BLOCK, BLOCK, 0, stream>>>(a, b, c, n);
}

__global__ static void k_scale_fp32(float* __restrict__ x, float scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= scale;
}

void scale_fp32(float* x, float scale, int n, cudaStream_t stream) {
    k_scale_fp32<<<(n+BLOCK-1)/BLOCK, BLOCK, 0, stream>>>(x, scale, n);
}

#define TILE 32
__global__ static void k_transpose_bf16(const __nv_bfloat16* __restrict__ src,
                                          __nv_bfloat16* __restrict__ dst,
                                          int M, int N) {
    __shared__ __nv_bfloat16 tile[TILE][TILE + 1];
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    if (x < N && y < M) tile[threadIdx.y][threadIdx.x] = src[y * N + x];
    __syncthreads();
    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;
    if (x < M && y < N) dst[y * M + x] = tile[threadIdx.x][threadIdx.y];
}

void transpose_bf16(const __nv_bfloat16* src, __nv_bfloat16* dst,
                    int M, int N, cudaStream_t stream) {
    dim3 block(TILE, TILE);
    dim3 grid((N+TILE-1)/TILE, (M+TILE-1)/TILE);
    k_transpose_bf16<<<grid, block, 0, stream>>>(src, dst, M, N);
}

} // namespace backend