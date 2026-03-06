// ============================================================
// FILE: trainer/include/backend/cuda_ops.h
// ============================================================
#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cstdint>

namespace backend {

// ---- cuBLAS matmul wrapper ----
// C = alpha * A @ B + beta * C
// A: [M, K]  B: [K, N]  C: [M, N]   (row-major)
// Supports bf16 inputs; accumulation in fp32 internally.
void gemm_bf16(cublasHandle_t   handle,
               int              M, int N, int K,
               float            alpha,
               const __nv_bfloat16* A,    // [M, K]
               const __nv_bfloat16* B,    // [K, N]
               float            beta,
               float*           C,        // [M, N]  fp32 output
               bool             transA = false,
               bool             transB = false);

// bf16 → bf16 gemm (output stays bf16, accumulated fp32 internally)
void gemm_bf16_out(cublasHandle_t   handle,
                   int              M, int N, int K,
                   const __nv_bfloat16* A,
                   const __nv_bfloat16* B,
                   __nv_bfloat16*   C,
                   bool             transA = false,
                   bool             transB = false);

// fp32 gemm
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

// SiLU: out[i] = x[i] / (1 + exp(-x[i]))   (bf16)
void silu_bf16(const __nv_bfloat16* x, __nv_bfloat16* out, int n, cudaStream_t s = 0);

// Fused SiLU-gating: out[i] = silu(gate[i]) * up[i]   (bf16)
void silu_mul_bf16(const __nv_bfloat16* gate,
                   const __nv_bfloat16* up,
                   __nv_bfloat16*       out,
                   int n, cudaStream_t s = 0);

// backward of silu_mul:
//   d_gate[i] = d_out[i] * up[i]  * silu'(gate[i])
//   d_up[i]   = d_out[i] * silu(gate[i])
// inputs are fp32 (gradients accumulate in fp32)
void silu_mul_backward_fp32(const float* d_out,     // [n]
                             const float* gate,      // [n]  fp32 gate pre-activation
                             const float* up,        // [n]  fp32 up activation
                             float*       d_gate,    // [n]
                             float*       d_up,      // [n]
                             int n, cudaStream_t s = 0);

// RMSNorm forward (bf16 input/output, fp32 weight)
//   out[b,i] = x[b,i] / rms(x[b]) * w[i]
//   rms[b] = sqrt(mean(x[b]^2) + eps)
void rmsnorm_bf16(const __nv_bfloat16* x,   // [rows, dim]
                  const __nv_bfloat16* w,   // [dim]
                  __nv_bfloat16*       out, // [rows, dim]
                  float*               rms, // [rows]  scratch: save for backward
                  int rows, int dim, float eps, cudaStream_t s = 0);

// RMSNorm backward  (fp32 gradients)
//   Given d_out [rows, dim] and saved rms [rows], x [rows, dim], w [dim]:
//   Compute d_x [rows, dim]
//   (d_w not computed here — not needed for frozen norms, handled separately if needed)
void rmsnorm_backward_fp32(const float*         d_out,  // [rows, dim]
                            const __nv_bfloat16* x,     // [rows, dim] original input
                            const __nv_bfloat16* w,     // [dim]
                            const float*         rms,   // [rows]
                            float*               d_x,   // [rows, dim]
                            int rows, int dim, cudaStream_t s = 0);

// Softmax in-place (fp32, row-wise)
void softmax_fp32(float* x, int rows, int cols, cudaStream_t s = 0);

// Softmax backward (fp32)
//   d_x[i] = p[i] * (d_out[i] - sum_j(d_out[j]*p[j]))
void softmax_backward_fp32(const float* p, const float* d_out,
                            float* d_x, int rows, int cols, cudaStream_t s = 0);

// bf16 → fp32 cast
void cast_bf16_to_fp32(const __nv_bfloat16* src, float* dst, int n, cudaStream_t s = 0);

// fp32 → bf16 cast
void cast_fp32_to_bf16(const float* src, __nv_bfloat16* dst, int n, cudaStream_t s = 0);

// Element-wise add (fp32)
void add_fp32(const float* a, const float* b, float* c, int n, cudaStream_t s = 0);

// Scale (fp32)
void scale_fp32(float* x, float s, int n, cudaStream_t s = 0);

// Transpose (bf16): [M, N] → [N, M]
void transpose_bf16(const __nv_bfloat16* src, __nv_bfloat16* dst,
                    int M, int N, cudaStream_t s = 0);

} // namespace backend