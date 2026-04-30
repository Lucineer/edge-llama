#ifndef EDGE_LLAMA_GGML_OPS_H
#define EDGE_LLAMA_GGML_OPS_H

#include <cstdint>
#include <vector>
#include <functional>

namespace edge_llama {

// =============================================================
// CPU-only tensor operations for Qwen2 inference
// No CUDA, no external dependencies beyond stdlib
// =============================================================

// Float types
struct block_q4_K {
    float d;           // scale
    float dmin;        // min scale
    uint8_t scales[12]; // combined scales
    uint8_t qs[128];    // 4-bit quants (2 per byte)
};

struct block_q6_K {
    float d;           // scale
    uint8_t ql[128];   // lower 4 bits
    uint8_t qh[64];    // upper 2 bits
    int8_t scales[32]; // 6-bit scales
};

// Dequantize Q4_K block → f32 array[256]
void dequantize_q4_K(const block_q4_K* block, float* out, int n = 256);

// Dequantize Q6_K block → f32 array[256]
void dequantize_q6_K(const block_q6_K* block, float* out);

// Dequantize F16 → f32
void dequantize_f16(const uint16_t* in, float* out, int n);

// =============================================================
// Matrix operations (row-major)
// =============================================================

// C = A @ B  (m×k  ·  k×n)
void matmul(const float* A, const float* B, float* C,
            int m, int k, int n);

// C = A @ B_T  (m×k  ·  n×k → m×n)
void matmul_trans_b(const float* A, const float* B, float* C,
                     int m, int k, int n);

// y = x @ W + b (generalized)
void linear(const float* x, const float* W, const float* bias,
            float* y, int in_dim, int out_dim);

// =============================================================
// Activation & normalization
// =============================================================

// SiLU (swish): y = x * sigmoid(x)
void silu(const float* x, float* y, int n);

// RMS Norm: y = x * rsqrt(mean(x^2) + eps) * weight
void rms_norm(const float* x, const float* weight, float* y,
              int n, float eps = 1e-6f);

// Softmax (in-place)
void softmax(float* x, int n);

// Element-wise add
void add(const float* a, const float* b, float* c, int n);

// Element-wise multiply
void mul(const float* a, const float* b, float* c, int n);

// RoPE (Rotary Position Embedding)
void apply_rope(float* q, float* k, int pos, int dim, int head_dim, float base);

// =============================================================
// Memory helpers
// =============================================================

// Aligned allocation for SIMD
float* aligned_alloc_f32(size_t n);
void aligned_free(float* p);

} // namespace edge_llama

#endif // EDGE_LLAMA_GGML_OPS_H
