#include "ggml_ops.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <cstdlib>

namespace edge_llama {

// Q4_K dequantization
// Each block encodes 256 values in 4-bit quantized + shared scale
void dequantize_q4_K(const block_q4_K* block, float* out, int n) {
    const uint16_t* sc = reinterpret_cast<const uint16_t*>(block->scales);
    const float d = block->d;
    const float min = block->dmin;
    
    // 6-bit scales: first 16 are in sc[0:7], next 16 spread in upper bits
    float scales[16];
    for (int i = 0; i < 8; i++) {
        scales[2*i]   = d * (sc[i] & 0xF);
        scales[2*i+1] = d * ((sc[i] >> 4) & 0xF);
        // This is simplified — actual Q4_K has more complex scale layout
    }
    // Apply high bits from scales[8:12] (not fully handled here)
    
    for (int i = 0; i < 128; i++) {
        float v = (block->qs[i] & 0xF) * scales[i % 16];
        out[i] = v - min;
        v = (block->qs[i] >> 4) * scales[i % 16];
        out[i + 128] = v - min;
    }
}

void dequantize_q6_K(const block_q6_K* block, float* out) {
    const float d = block->d;
    for (int i = 0; i < 128; i++) {
        int low = block->ql[i];
        int high_low = block->qh[i / 2] >> (4 * (i % 2));
        int high = (high_low & 0x0F) << 4; // upper 2 bits of 6-bit value
        int val = (low & 0x0F) | (high & 0x30); // first nibble
        out[i] = d * (val - block->scales[i % 32]);
        
        val = (low >> 4) | ((high >> 2) & 0x30); // second nibble
        out[i + 128] = d * (val - block->scales[i % 32]);
    }
}

void dequantize_f16(const uint16_t* in, float* out, int n) {
    for (int i = 0; i < n; i++) {
        uint16_t h = in[i];
        // FP16 to FP32
        uint32_t sign = (h & 0x8000) << 16;
        uint32_t exp = (h & 0x7C00) >> 10;
        uint32_t mant = (h & 0x03FF);
        
        if (exp == 0) {
            // Subnormal
            float v;
            uint32_t bits = sign | (112 << 23) | mant;
            std::memcpy(&v, &bits, sizeof(v));
            out[i] = v - 0.5f; // denormal bias
        } else if (exp == 31) {
            // NaN or Inf
            uint32_t bits = sign | 0x7F800000 | (mant << 13);
            std::memcpy(&out[i], &bits, sizeof(out[i]));
        } else {
            uint32_t bits = sign | ((exp + 112) << 23) | (mant << 13);
            std::memcpy(&out[i], &bits, sizeof(out[i]));
        }
    }
}

void matmul(const float* A, const float* B, float* C, int m, int k, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int l = 0; l < k; l++) {
                sum += A[i * k + l] * B[l * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

void matmul_trans_b(const float* A, const float* B, float* C, int m, int k, int n) {
    // B is (n × k), we need (m × k) · (k × n) → but B is stored as n,k and needs transposition
    // Actually if B is shape (n, k), we want C[m,n] = A[m,k] @ B^T[n,k]
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int l = 0; l < k; l++) {
                sum += A[i * k + l] * B[j * k + l]; // B[j][l]
            }
            C[i * n + j] = sum;
        }
    }
}

void linear(const float* x, const float* W, const float* bias,
            float* y, int in_dim, int out_dim) {
    for (int j = 0; j < out_dim; j++) {
        float sum = bias ? bias[j] : 0.0f;
        for (int i = 0; i < in_dim; i++) {
            sum += x[i] * W[j * in_dim + i];
        }
        y[j] = sum;
    }
}

void silu(const float* x, float* y, int n) {
    for (int i = 0; i < n; i++) {
        y[i] = x[i] / (1.0f + std::exp(-x[i]));
    }
}

void rms_norm(const float* x, const float* weight, float* y, int n, float eps) {
    float sum_sq = 0.0f;
    for (int i = 0; i < n; i++) {
        sum_sq += x[i] * x[i];
    }
    float rms = std::sqrt(sum_sq / n + eps);
    float inv_rms = 1.0f / rms;
    for (int i = 0; i < n; i++) {
        y[i] = x[i] * inv_rms * weight[i];
    }
}

void softmax(float* x, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; i++) if (x[i] > max_val) max_val = x[i];
    
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = std::exp(x[i] - max_val);
        sum += x[i];
    }
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < n; i++) x[i] *= inv_sum;
}

void add(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; i++) c[i] = a[i] + b[i];
}

void mul(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; i++) c[i] = a[i] * b[i];
}

void apply_rope(float* q, float* k, int pos, int dim, int head_dim, float base) {
    // Apply rotary embeddings to q and k for a given position
    for (int i = 0; i < dim; i += 2) {
        float theta = std::pow(base, -2.0f * i / head_dim) * pos;
        float cos_t = std::cos(theta);
        float sin_t = std::sin(theta);
        
        float q0 = q[i];
        float q1 = q[i + 1];
        q[i] = q0 * cos_t - q1 * sin_t;
        q[i + 1] = q0 * sin_t + q1 * cos_t;
        
        float k0 = k[i];
        float k1 = k[i + 1];
        k[i] = k0 * cos_t - k1 * sin_t;
        k[i + 1] = k0 * sin_t + k1 * cos_t;
    }
}

float* aligned_alloc_f32(size_t n) {
    void* p;
    if (posix_memalign(&p, 64, n * sizeof(float)) != 0) return nullptr;
    return static_cast<float*>(p);
}

void aligned_free(float* p) {
    free(p);
}

} // namespace edge_llama
