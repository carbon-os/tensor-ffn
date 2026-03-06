// ============================================================
// FILE: trainer/src/training/loss.cpp
// ============================================================
#include "training/loss.h"
#include <cuda_runtime.h>
#include <cstdio>

namespace training {

// Cross-entropy kernel:
//   For each token t: loss[t] = -log(softmax(logits[t])[target[t]])
//   d_logits[t, v] = softmax(logits[t])[v] - (v == target[t] ? 1 : 0)
// We do this in a fused way: softmax then subtract 1 at target.

__global__ static void k_cross_entropy(const float* logits,  // [T, V]
                                        const int32_t* targets, // [T]
                                        float* d_logits,       // [T, V]
                                        float* losses,         // [T]
                                        int T, int V) {
    // Each block handles one token position
    int t = blockIdx.x;
    if (t >= T) return;
    const float* lg = logits  + t * V;
    float*       dlg = d_logits + t * V;
    int tgt = targets[t];

    // Find max for numerical stability
    float maxv = -1e30f;
    for (int v = threadIdx.x; v < V; v += blockDim.x) maxv = fmaxf(maxv, lg[v]);
    __shared__ float smx[256];
    smx[threadIdx.x] = maxv;
    __syncthreads();
    for (int s = blockDim.x>>1; s > 0; s >>= 1) {
        if (threadIdx.x < s) smx[threadIdx.x] = fmaxf(smx[threadIdx.x], smx[threadIdx.x+s]);
        __syncthreads();
    }
    maxv = smx[0];

    // Compute softmax and accumulate sum
    float sumv = 0.0f;
    for (int v = threadIdx.x; v < V; v += blockDim.x) {
        float e = expf(lg[v] - maxv);
        dlg[v] = e;  // temporarily store exp(logit)
        sumv += e;
    }
    __shared__ float ssum[256];
    ssum[threadIdx.x] = sumv;
    __syncthreads();
    for (int s = blockDim.x>>1; s > 0; s >>= 1) {
        if (threadIdx.x < s) ssum[threadIdx.x] += ssum[threadIdx.x+s];
        __syncthreads();
    }
    sumv = ssum[0];

    // Normalise → softmax, then compute d_logits = (softmax - one_hot) / T
    float inv_sum = 1.0f / sumv;
    float log_prob = 0.0f;
    for (int v = threadIdx.x; v < V; v += blockDim.x) {
        float p = dlg[v] * inv_sum;
        dlg[v] = p / (float)T;  // divide by T for mean loss backward
        if (v == tgt) {
            dlg[v] -= 1.0f / (float)T;
            log_prob = logf(fmaxf(p, 1e-10f));
        }
    }
    // Accumulate loss at target thread
    if (threadIdx.x == 0) {
        // Collect log_prob from all threads (only one thread has it)
        // Use atomics or single-thread approach. Since only one v==tgt, one thread does this.
        // Simpler: reduce across threads
    }
    // Parallel reduction to find log_prob (one thread set it)
    __shared__ float slp[256];
    slp[threadIdx.x] = log_prob;
    __syncthreads();
    for (int s = blockDim.x>>1; s > 0; s >>= 1) {
        if (threadIdx.x < s) slp[threadIdx.x] += slp[threadIdx.x+s];
        __syncthreads();
    }
    if (threadIdx.x == 0) losses[t] = -slp[0];
}

// Reduce losses array to scalar on device, copy to host
__global__ static void k_mean(const float* losses, float* out, int T) {
    float s = 0.0f;
    for (int i = threadIdx.x; i < T; i += blockDim.x) s += losses[i];
    __shared__ float ss[256];
    ss[threadIdx.x] = s;
    __syncthreads();
    for (int d = blockDim.x>>1; d > 0; d >>= 1) {
        if (threadIdx.x < d) ss[threadIdx.x] += ss[threadIdx.x+d];
        __syncthreads();
    }
    if (threadIdx.x == 0) *out = ss[0] / T;
}

float cross_entropy_loss(cublasHandle_t /*cublas*/,
                          const float* logits,
                          const int32_t* targets,
                          float* d_logits,
                          int T, int V,
                          cudaStream_t s) {
    float* losses_d;
    cudaMalloc(&losses_d, T * sizeof(float));
    float* mean_d;
    cudaMalloc(&mean_d, sizeof(float));

    k_cross_entropy<<<T, 256, 0, s>>>(logits, targets, d_logits, losses_d, T, V);
    k_mean<<<1, 256, 0, s>>>(losses_d, mean_d, T);

    float mean_h;
    cudaMemcpyAsync(&mean_h, mean_d, sizeof(float), cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);

    cudaFree(losses_d);
    cudaFree(mean_d);
    return mean_h;
}

} // namespace training