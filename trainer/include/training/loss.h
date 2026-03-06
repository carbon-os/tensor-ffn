// ============================================================
// FILE: trainer/include/training/loss.h
// ============================================================
#pragma once
#include <cublas_v2.h>

namespace training {

// Cross-entropy loss over all token positions.
// logits:  [T, V]  fp32 device  (T = batch * seq)
// targets: [T]     int32 device
// Returns: mean scalar loss (device).
// Writes d_logits [T, V] fp32 device — normalised by T.
float cross_entropy_loss(cublasHandle_t  cublas,
                         const float*    logits,    // [T, V]
                         const int32_t*  targets,   // [T]
                         float*          d_logits,  // [T, V]  output
                         int             T,
                         int             V,
                         cudaStream_t    s = 0);

} // namespace training