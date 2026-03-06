#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cstdint>

namespace backend {

// ---- cuBLAS matmul wrappers ----
// C = alpha * A @ B + beta * C
// A: [M, K]  B: [K, N]  C: [M, N]  (row-major)

// bf16 inputs → fp32 output
void gemm_bf16(cublasHandle_t       handle,
               int                  M, int N, int K,
               float                alpha,
               const __nv_bfloat16* A,
               const __nv_bfloat16* B,
               float                beta,
               float*               C,
               bool                 transA = false,
               bool                 transB = false);

// bf16 inputs → bf16 output
void gemm_bf16_out(cublasHandle_t       handle,
                   int                  M, int N, int K,
                   const __nv_bfloat16* A,
                   const __nv_bfloat16* B,
                   __nv_bfloat16*       C,
                   bool                 transA = false,
                   bool                 transB = false);

// fp32 → fp32
void gemm_fp32(cublasHandle_t handle,
               int            M, int N, int K,
               float          alpha,
               const float*   A,
               const float*   B,
               float          beta,
               float*         C,
               bool           transA = false,
               bool           transB = false);

// ---- element-wise kernels ----

// SiLU: out[i] = x[i] / (1 + exp(-x[i]))  (bf16)
void silu_bf16(const __nv_bfloat16* x,
               __nv_bfloat16*       out,
               int                  n,
               cudaStream_t         stream = 0);

// Fused SiLU-gating: out[i] = silu(gate[i]) * up[i]  (bf16)
void silu_mul_bf16(const __nv_bfloat16* gate,
                   const __nv_bfloat16* up,
                   __nv_bfloat16*       out,
                   int                  n,
                   cudaStream_t         stream = 0);

// Backward of silu_mul (fp32 gradients):
//   d_gate[i] = d_out[i] * up[i] * silu'(gate[i])
//   d_up[i]   = d_out[i] * silu(gate[i])
void silu_mul_backward_fp32(const float* d_out,
                             const float* gate,
                             const float* up,
                             float*       d_gate,
                             float*       d_up,
                             int          n,
                             cudaStream_t stream = 0);

// RMSNorm forward (bf16 input/output, saves rms for backward)
//   out[b,i] = x[b,i] / rms(x[b]) * w[i]
void rmsnorm_bf16(const __nv_bfloat16* x,
                  const __nv_bfloat16* w,
                  __nv_bfloat16*       out,
                  float*               rms,
                  int                  rows,
                  int                  dim,
                  float                eps,
                  cudaStream_t         stream = 0);

// RMSNorm backward (fp32 gradients)
void rmsnorm_backward_fp32(const float*         d_out,
                            const __nv_bfloat16* x,
                            const __nv_bfloat16* w,
                            const float*         rms,
                            float*               d_x,
                            int                  rows,
                            int                  dim,
                            cudaStream_t         stream = 0);

// Softmax in-place (fp32, row-wise)
void softmax_fp32(float*       x,
                  int          rows,
                  int          cols,
                  cudaStream_t stream = 0);

// Softmax backward (fp32)
void softmax_backward_fp32(const float* p,
                            const float* d_out,
                            float*       d_x,
                            int          rows,
                            int          cols,
                            cudaStream_t stream = 0);

// bf16 → fp32 cast
void cast_bf16_to_fp32(const __nv_bfloat16* src,
                        float*               dst,
                        int                  n,
                        cudaStream_t         stream = 0);

// fp32 → bf16 cast
void cast_fp32_to_bf16(const float*   src,
                        __nv_bfloat16* dst,
                        int            n,
                        cudaStream_t   stream = 0);

// Element-wise add (fp32): c[i] = a[i] + b[i]
void add_fp32(const float* a,
              const float* b,
              float*       c,
              int          n,
              cudaStream_t stream = 0);

// Scale in-place (fp32): x[i] *= scale
void scale_fp32(float*       x,
                float        scale,
                int          n,
                cudaStream_t stream = 0);

// Transpose (bf16): [M, N] → [N, M]
void transpose_bf16(const __nv_bfloat16* src,
                    __nv_bfloat16*       dst,
                    int                  M,
                    int                  N,
                    cudaStream_t         stream = 0);

} // namespace backend