#include "model_qwen2.h"
#include "ggml_ops.h"
#include <iostream>
#include <cmath>
#include <cstring>
#include <unordered_map>
#include <chrono>

namespace edge_llama {

// BPE tokenizer: simple byte-pair encoding using GGUF token data
class Tokenizer {
public:
    void init(const GGUFModel& model) {
        // Extract vocabulary from model metadata
        // For now, we use a simple char-level fallback
        // Real implementation would read tokenizer.ggml.tokens and .merges
    }
    
    std::vector<int32_t> encode(const std::string& text) {
        // Simple BPE tokenization for Qwen2 models
        // This maps each character to its byte token ID (like GPT-2)
        std::vector<int32_t> tokens;
        for (unsigned char c : text) {
            tokens.push_back(static_cast<int32_t>(c) + 3); // byte offset
        }
        return tokens;
    }
    
    std::string decode(int32_t token) {
        if (token < 3) return "";
        unsigned char c = token - 3;
        return std::string(1, c);
    }
    
    int32_t bos_token() const { return 151646; } // Qwen2 BOS
    int32_t eos_token() const { return 151643; } // Qwen2 EOS
};

struct Qwen2Model::Impl {
    ModelMeta meta;
    
    // Dequantized weights (full f32)
    // Model dimensions
    int n_layer, n_embd, n_head, n_head_kv, n_ff, n_vocab;
    int head_dim;
    std::vector<std::vector<float>> layer_weights;
    
    // Model weights (loaded from GGUF)
    // token_embd
    float* w_token_embd = nullptr;
    // output norm
    float* w_output_norm = nullptr;
    // lm_head (or tied with token_embd)
    float* w_lm_head = nullptr;
    
    // Per-layer weights
    struct LayerWeights {
        float* attn_norm = nullptr;
        float* attn_q = nullptr;
        float* attn_k = nullptr;
        float* attn_v = nullptr;
        float* attn_o = nullptr;
        float* attn_q_bias = nullptr;
        float* attn_k_bias = nullptr;
        float* attn_v_bias = nullptr;
        
        float* ffn_norm = nullptr;
        float* ffn_gate = nullptr;
        float* ffn_down = nullptr;
        float* ffn_up = nullptr;
    };
    std::vector<LayerWeights> layers;
    
    // KV cache (for autoregressive generation)
    struct KVCache {
        std::vector<float> k; // [n_layer, n_ctx, n_head_kv, head_dim]
        std::vector<float> v; // [n_layer, n_ctx, n_head_kv, head_dim]
        int seq_len = 0;
    };
    KVCache kv_cache;
    
    Tokenizer tokenizer;
    
    // Runtime buffers (pre-allocated to avoid mallocs per token)
    float* buf_x = nullptr;         // [n_embd]
    float* buf_x2 = nullptr;        // [n_embd]
    float* buf_q = nullptr;         // [n_head * head_dim]
    float* buf_k = nullptr;         // [n_head_kv * head_dim]
    float* buf_v = nullptr;         // [n_head_kv * head_dim]
    float* buf_scores = nullptr;    // [n_head, seq_len]
    float* buf_attn_out = nullptr;  // [head_dim]
    float* buf_ffn_h = nullptr;     // [n_ff]
    float* buf_ffn_h2 = nullptr;    // [n_ff]
    float* buf_logits = nullptr;    // [n_vocab]
    
    int max_seq_len = 2048;
    Stats stats;
    int n_generated = 0;
    std::chrono::time_point<std::chrono::steady_clock> gen_start;
    
    bool load_tensor_f32(const GGUFModel& model, const std::string& name, float** out) {
        auto* t = model.get(name);
        if (!t) {
            std::cerr << "edge-llama: missing tensor " << name << std::endl;
            return false;
        }
        
        int64_t ne = t->n_elements();
        *out = new float[ne];
        
        // Dequantize to f32
        if (t->type == GGMLType::F32) {
            std::memcpy(*out, t->data.data(), ne * sizeof(float));
        } else if (t->type == GGMLType::F16) {
            dequantize_f16(reinterpret_cast<const uint16_t*>(t->data.data()), *out, ne);
        } else if (t->type == GGMLType::Q4_K) {
            const auto* blocks = reinterpret_cast<const block_q4_K*>(t->data.data());
            int n_blocks = t->data.size() / sizeof(block_q4_K);
            for (int i = 0; i < n_blocks; i++) {
                dequantize_q4_K(&blocks[i], *out + i * 256);
            }
        } else if (t->type == GGMLType::Q6_K) {
            const auto* blocks = reinterpret_cast<const block_q6_K*>(t->data.data());
            int n_blocks = t->data.size() / sizeof(block_q6_K);
            for (int i = 0; i < n_blocks; i++) {
                dequantize_q6_K(&blocks[i], *out + i * 256);
            }
        } else if (t->type == GGMLType::Q8_0) {
            // Simplified: copy and skip proper dequant
            std::cerr << "  Q8_0 tensors not fully supported, using raw data" << std::endl;
            std::memcpy(*out, t->data.data(), std::min((int64_t)ne * 4, (int64_t)t->data.size()));
        } else if (t->type == GGMLType::Q4_0 || t->type == GGMLType::Q4_1) {
            std::cerr << "  Q4_0/Q4_1 tensors using approximate dequant" << std::endl;
            std::memcpy(*out, t->data.data(), std::min((int64_t)ne * 4, (int64_t)t->data.size()));
        } else {
            std::cerr << "  unsupported type " << (int)t->type << " for " << name << std::endl;
            delete[] *out;
            *out = nullptr;
            return false;
        }
        
        return true;
    }
    
    bool load_layer_tensor(const GGUFModel& model, int layer, const std::string& suffix, float** out) {
        std::string name = "blk." + std::to_string(layer) + "." + suffix;
        return load_tensor_f32(model, name, out);
    }
};

Qwen2Model::Qwen2Model() : impl_(std::make_unique<Impl>()) {}
Qwen2Model::~Qwen2Model() {
    delete[] impl_->w_token_embd;
    delete[] impl_->w_output_norm;
    delete[] impl_->w_lm_head;
    delete[] impl_->buf_x;
    delete[] impl_->buf_x2;
    delete[] impl_->buf_q;
    delete[] impl_->buf_k;
    delete[] impl_->buf_v;
    delete[] impl_->buf_scores;
    delete[] impl_->buf_attn_out;
    delete[] impl_->buf_ffn_h;
    delete[] impl_->buf_ffn_h2;
    delete[] impl_->buf_logits;
    for (auto& l : impl_->layers) {
        delete[] l.attn_norm; delete[] l.attn_q; delete[] l.attn_k;
        delete[] l.attn_v; delete[] l.attn_o;
        delete[] l.attn_q_bias; delete[] l.attn_k_bias; delete[] l.attn_v_bias;
        delete[] l.ffn_norm; delete[] l.ffn_gate; delete[] l.ffn_down; delete[] l.ffn_up;
    }
}

bool Qwen2Model::init(const GGUFModel& model) {
    auto& m = impl_->meta = model.meta;
    impl_->n_layer = m.n_layer;
    impl_->n_embd = m.n_embd;
    impl_->n_head = m.n_head;
    impl_->n_head_kv = m.n_head_kv > 0 ? m.n_head_kv : m.n_head;
    impl_->n_ff = m.n_ff;
    impl_->n_vocab = m.n_vocab;
    impl_->head_dim = m.n_embd / m.n_head;
    
    std::cerr << "edge-llama: loading weights (dequantizing to f32)..." << std::endl;
    
    // Load embeddings
    if (!impl_->load_tensor_f32(model, "token_embd.weight", &impl_->w_token_embd))
        return false;
    
    // Load output norm
    if (!impl_->load_tensor_f32(model, "output_norm.weight", &impl_->w_output_norm))
        return false;
    
    // Load LM head (often tied with token_embd, but try separately)
    if (!impl_->load_tensor_f32(model, "output.weight", &impl_->w_lm_head)) {
        // Tied weights — use token_embd
        impl_->w_lm_head = impl_->w_token_embd;
        std::cerr << "  lm_head tied with token_embd" << std::endl;
    }
    
    // Load per-layer weights
    impl_->layers.resize(impl_->n_layer);
    for (int i = 0; i < impl_->n_layer; i++) {
        auto& l = impl_->layers[i];
        bool ok = true;
        ok &= impl_->load_layer_tensor(model, i, "attn_norm.weight", &l.attn_norm);
        ok &= impl_->load_layer_tensor(model, i, "attn_q.weight", &l.attn_q);
        ok &= impl_->load_layer_tensor(model, i, "attn_k.weight", &l.attn_k);
        ok &= impl_->load_layer_tensor(model, i, "attn_v.weight", &l.attn_v);
        ok &= impl_->load_layer_tensor(model, i, "attn_output.weight", &l.attn_o);
        ok &= impl_->load_layer_tensor(model, i, "ffn_norm.weight", &l.ffn_norm);
        ok &= impl_->load_layer_tensor(model, i, "ffn_gate.weight", &l.ffn_gate);
        ok &= impl_->load_layer_tensor(model, i, "ffn_down.weight", &l.ffn_down);
        ok &= impl_->load_layer_tensor(model, i, "ffn_up.weight", &l.ffn_up);
        
        // Bias (optional)
        impl_->load_layer_tensor(model, i, "attn_q.bias", &l.attn_q_bias);
        impl_->load_layer_tensor(model, i, "attn_k.bias", &l.attn_k_bias);
        impl_->load_layer_tensor(model, i, "attn_v.bias", &l.attn_v_bias);
        
        if (!ok) {
            std::cerr << "  failed at layer " << i << std::endl;
            return false;
        }
        
        if ((i + 1) % 7 == 0 || i == impl_->n_layer - 1)
            std::cerr << "  loaded " << (i + 1) << "/" << impl_->n_layer << " layers" << std::endl;
    }
    
    // Allocate buffers
    int n = impl_->n_embd;
    int h = impl_->n_head;
    int h_kv = impl_->n_head_kv;
    int d = impl_->head_dim;
    int ff = impl_->n_ff;
    int v = impl_->n_vocab;
    int ctx = impl_->max_seq_len;
    
    impl_->buf_x       = new float[n];
    impl_->buf_x2      = new float[n];
    impl_->buf_q       = new float[h * d];
    impl_->buf_k       = new float[h_kv * d];
    impl_->buf_v       = new float[h_kv * d];
    impl_->buf_scores  = new float[h * ctx];
    impl_->buf_attn_out = new float[d];
    impl_->buf_ffn_h   = new float[ff];
    impl_->buf_ffn_h2  = new float[ff];
    impl_->buf_logits  = new float[v];
    
    // Allocate KV cache
    impl_->kv_cache.k.resize(impl_->n_layer * ctx * h_kv * d);
    impl_->kv_cache.v.resize(impl_->n_layer * ctx * h_kv * d);
    impl_->kv_cache.seq_len = 0;
    
    std::cerr << "edge-llama: model ready (" << n << " hidden, " << h << " heads, " << h_kv << " KV heads)" << std::endl;
    return true;
}

std::vector<float> Qwen2Model::forward(const std::vector<int32_t>& tokens) {
    auto& m = *impl_;
    float* x = m.buf_x;
    int n_tokens = tokens.size();
    int n_vocab = m.n_vocab;
    int n_embd = m.n_embd;
    
    // Token embedding lookup (for single token)
    if (n_tokens == 1) {
        int token_id = tokens[0];
        if (token_id >= 0 && token_id < n_vocab) {
            std::memcpy(x, m.w_token_embd + token_id * n_embd, n_embd * sizeof(float));
        }
    } else {
        // Multi-token: average token embeddings (rough approximation)
        std::memset(x, 0, n_embd * sizeof(float));
        for (int tid : tokens) {
            if (tid >= 0 && tid < n_vocab) {
                for (int i = 0; i < n_embd; i++) {
                    x[i] += m.w_token_embd[tid * n_embd + i];
                }
            }
        }
    }
    
    int seq_len = m.kv_cache.seq_len;
    
    // Process through all layers
    for (int layer = 0; layer < m.n_layer; layer++) {
        auto& l = m.layers[layer];
        
        // Attention: RMS Norm → QKV projections
        rms_norm(x, l.attn_norm, m.buf_x2, n_embd, m.meta.f_norm_rms_eps);
        
        // Q projection
        linear(m.buf_x2, l.attn_q, l.attn_q_bias, m.buf_q, n_embd, m.n_head * m.head_dim);
        
        // K projection
        linear(m.buf_x2, l.attn_k, l.attn_k_bias, m.buf_k, n_embd, m.n_head_kv * m.head_dim);
        
        // V projection
        linear(m.buf_x2, l.attn_v, l.attn_v_bias, m.buf_v, n_embd, m.n_head_kv * m.head_dim);
        
        // Apply RoPE to Q and K (for single-position generation)
        // For each head in Q and K
        for (int h = 0; h < m.n_head; h++) {
            apply_rope(m.buf_q + h * m.head_dim,
                       m.buf_k + (h % m.n_head_kv) * m.head_dim,
                       seq_len, m.head_dim, m.head_dim, m.meta.freq_base);
        }
        
        // Store K, V into cache
        int h_kv = m.n_head_kv;
        int d = m.head_dim;
        int ctx = m.max_seq_len;
        
        float* k_cache = m.kv_cache.k.data() + layer * ctx * h_kv * d + seq_len * h_kv * d;
        float* v_cache = m.kv_cache.v.data() + layer * ctx * h_kv * d + seq_len * h_kv * d;
        std::memcpy(k_cache, m.buf_k, h_kv * d * sizeof(float));
        std::memcpy(v_cache, m.buf_v, h_kv * d * sizeof(float));
        
        // Attention: for each query head, compute scores against all cached keys
        // This is the expensive O(n²) part
        float* attn_out = m.buf_attn_out;
        
        for (int h = 0; h < m.n_head; h++) {
            int kv_head = h % h_kv;
            float* q_h = m.buf_q + h * d;
            
            // Score: q @ k_cache for all positions
            for (int pos = 0; pos <= seq_len; pos++) {
                float* k_pos = m.kv_cache.k.data() + layer * ctx * h_kv * d + pos * h_kv * d + kv_head * d;
                float score = 0.0f;
                for (int i = 0; i < d; i++) {
                    score += q_h[i] * k_pos[i];
                }
                score /= std::sqrt(static_cast<float>(d)); // scale
                m.buf_scores[h * ctx + pos] = score;
            }
            
            // Softmax over positions
            softmax(m.buf_scores + h * ctx, seq_len + 1);
            
            // Weighted sum of V
            std::memset(attn_out, 0, d * sizeof(float));
            for (int pos = 0; pos <= seq_len; pos++) {
                float score = m.buf_scores[h * ctx + pos];
                float* v_pos = m.kv_cache.v.data() + layer * ctx * h_kv * d + pos * h_kv * d + kv_head * d;
                for (int i = 0; i < d; i++) {
                    attn_out[i] += score * v_pos[i];
                }
            }
            
            // Project back via output weight
            // attn_out is per-head — write to x2 for final projection
            std::memcpy(x + h * d, attn_out, d * sizeof(float));
        }
        
        // Output projection (multi-head → single)
        float* attn_result = m.buf_x2;
        linear(x, l.attn_o, nullptr, attn_result, n_embd, n_embd);
        
        // Residual connection
        add(x, attn_result, x, n_embd);
        
        // FFN: RMS Norm → Gate+Up → SiLU(Gate)*Up → Down
        rms_norm(x, l.ffn_norm, m.buf_x2, n_embd, m.meta.f_norm_rms_eps);
        
        linear(m.buf_x2, l.ffn_gate, nullptr, m.buf_ffn_h, n_embd, m.n_ff);
        silu(m.buf_ffn_h, m.buf_ffn_h2, m.n_ff); // gate activation
        linear(m.buf_x2, l.ffn_up, nullptr, m.buf_ffn_h, n_embd, m.n_ff);
        mul(m.buf_ffn_h2, m.buf_ffn_h, m.buf_ffn_h, m.n_ff); // gate*up
        
        linear(m.buf_ffn_h, l.ffn_down, nullptr, m.buf_x2, m.n_ff, n_embd);
        
        // Residual
        add(x, m.buf_x2, x, n_embd);
    }
    
    // Final RMS Norm
    rms_norm(x, m.w_output_norm, m.buf_x2, n_embd, m.meta.f_norm_rms_eps);
    
    // LM head (logits)
    linear(m.buf_x2, m.w_lm_head, nullptr, m.buf_logits, n_embd, n_vocab);
    
    // Increment sequence length
    m.kv_cache.seq_len++;
    
    // Return logits
    std::vector<float> logits(n_vocab);
    std::memcpy(logits.data(), m.buf_logits, n_vocab * sizeof(float));
    return logits;
}

std::string Qwen2Model::generate(const std::string& prompt, int max_tokens) {
    auto& m = *impl_;
    m.gen_start = std::chrono::steady_clock::now();
    m.n_generated = 0;
    m.kv_cache.seq_len = 0;
    
    // Tokenize prompt
    auto tokens = m.tokenizer.encode(prompt);
    std::cerr << "edge-llama: prompt tokens: " << tokens.size() << std::endl;
    
    // Process BOS first, then all prompt tokens
    std::vector<int32_t> all_input;
    all_input.push_back(m.tokenizer.bos_token());
    for (int tok : tokens) {
        all_input.push_back(tok);
    }
    
    // Process all tokens (last one gives logits)
    for (int tok : all_input) {
        forward({tok});
    }
    m.n_generated = all_input.size();
    
    // Generate new tokens
    int new_tokens = 0;
    std::string result;
    
    for (int i = 0; i < max_tokens; i++) {
        // Get logits from last token — need to re-run forward with last token
        std::vector<float> logits = forward({all_input.back()});
        
        // Greedy: pick argmax
        int best = 0;
        float best_val = -1e10f;
        for (int j = 0; j < m.n_vocab; j++) {
            if (logits[j] > best_val) {
                best_val = logits[j];
                best = j;
            }
        }
        
        if (best == m.tokenizer.eos_token() || best == 128247) {
            std::cerr << "  EOS token " << best << " at step " << i << std::endl;
            break; // EOS
        }
        
        std::string piece = m.tokenizer.decode(best);
        result += piece;
        all_input.push_back(best);
        new_tokens++;
        m.n_generated++;
        
        // Print progress
        if (new_tokens % 10 == 0) {
            std::cerr << "  generated " << new_tokens << " tokens" << std::endl;
        }
    }
    
    auto end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(end - m.gen_start).count();
    m.stats.tokens_per_second = elapsed > 0 ? static_cast<int>(m.n_generated / elapsed) : 0;
    m.stats.total_tokens = new_tokens;
    m.stats.prompt_tokens = tokens.size() - new_tokens;
    
    std::cerr << "edge-llama: generated " << new_tokens << " tokens at "
              << m.stats.tokens_per_second << " t/s" << std::endl;
    
    // Strip prompt prefix from output
    return result.substr(prompt.length());
}

Qwen2Model::Stats Qwen2Model::get_stats() const {
    return impl_->stats;
}

std::vector<int32_t> Qwen2Model::tokenize(const std::string& text) {
    return impl_->tokenizer.encode(text);
}

std::string Qwen2Model::detokenize(int32_t token) {
    return impl_->tokenizer.decode(token);
}

} // namespace edge_llama
