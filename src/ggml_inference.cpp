#include "ggml_inference.h"
#include <iostream>
#include <chrono>
#include <cmath>
#include <cstring>
#include <vector>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

namespace edge_llama {

// BPE tokenizer (simplified)
class Tokenizer {
public:
    void init() {}
    
    std::vector<int32_t> encode(const std::string& text) {
        std::vector<int32_t> tokens;
        for (unsigned char c : text) {
            tokens.push_back(static_cast<int32_t>(c) + 3);
        }
        return tokens;
    }
    
    std::string decode(int32_t token) {
        if (token < 3) return "";
        if (token >= 151936) return "";
        unsigned char c = token - 3;
        return std::string(1, c);
    }
    
    int32_t bos_token() const { return 151646; }
    int32_t eos_token() const { return 151643; }
};

struct GGMLInference::Impl {
    // Model metadata
    int n_layer, n_embd, n_head, n_head_kv, n_ff, n_vocab, head_dim;
    ModelMeta meta;
    
    // ggml backend
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_type_t buft = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    
    // Weights as ggml tensors
    // These are reference tensors — actual data is stored separately
    struct WeightTensors {
        ggml_tensor* token_embd = nullptr;
        ggml_tensor* output_norm = nullptr;
        ggml_tensor* output = nullptr;
        
        struct Layer {
            ggml_tensor* attn_norm = nullptr;
            ggml_tensor* attn_q = nullptr;
            ggml_tensor* attn_k = nullptr;
            ggml_tensor* attn_v = nullptr;
            ggml_tensor* attn_o = nullptr;
            ggml_tensor* attn_q_b = nullptr;
            ggml_tensor* attn_k_b = nullptr;
            ggml_tensor* attn_v_b = nullptr;
            ggml_tensor* ffn_norm = nullptr;
            ggml_tensor* ffn_gate = nullptr;
            ggml_tensor* ffn_down = nullptr;
            ggml_tensor* ffn_up = nullptr;
        };
        std::vector<Layer> layers;
    };
    WeightTensors w;
    
    // Raw weight data (we manage lifetime)
    std::vector<uint8_t> weight_data;
    
    // ggml compute context
    ggml_context* compute_ctx = nullptr;
    
    // KV cache
    struct KVCell {
        std::vector<float> k; // [n_layer, n_ctx, n_head_kv * head_dim]
        std::vector<float> v;
        int seq_len = 0;
    };
    KVCell kv;
    
    Tokenizer tokenizer;
    Stats stats;
    std::chrono::time_point<std::chrono::steady_clock> gen_start;
    
    // Get ggml type from GGMLType
    static ggml_type to_ggml_type(GGMLType t) {
        switch (t) {
            case GGMLType::F32:  return GGML_TYPE_F32;
            case GGMLType::F16:  return GGML_TYPE_F16;
            case GGMLType::Q4_0: return GGML_TYPE_Q4_0;
            case GGMLType::Q4_1: return GGML_TYPE_Q4_1;
            case GGMLType::Q5_0: return GGML_TYPE_Q5_0;
            case GGMLType::Q5_1: return GGML_TYPE_Q5_1;
            case GGMLType::Q8_0: return GGML_TYPE_Q8_0;
            case GGMLType::Q2_K: return GGML_TYPE_Q2_K;
            case GGMLType::Q3_K: return GGML_TYPE_Q3_K;
            case GGMLType::Q4_K: return GGML_TYPE_Q4_K;
            case GGMLType::Q5_K: return GGML_TYPE_Q5_K;
            case GGMLType::Q6_K: return GGML_TYPE_Q6_K;
            default: return GGML_TYPE_F32;
        }
    }
    
    // Load raw tensor data into a ggml tensor
    void load_tensor(ggml_context* ctx, const GGUFModel& src, 
                     const std::string& name, ggml_tensor** dst) {
        auto* t = src.get(name);
        if (!t) { *dst = nullptr; return; }
        
        auto gtype = to_ggml_type(t->type);
        
        // Create weight tensor
        if (t->dims.size() == 2) {
            *dst = ggml_new_tensor_2d(ctx, gtype, t->dims[0], t->dims[1]);
        } else if (t->dims.size() == 1) {
            *dst = ggml_new_tensor_1d(ctx, gtype, t->dims[0]);
        } else if (t->dims.size() == 0) {
            *dst = ggml_new_tensor_1d(ctx, gtype, 1);
        } else {
            *dst = nullptr;
            return;
        }
        
        if (*dst) {
            // Copy raw quantization data directly
            std::memcpy((*dst)->data, t->data.data(), t->data.size());
        }
    }
};

GGMLInference::GGMLInference() : impl_(std::make_unique<Impl>()) {}
GGMLInference::~GGMLInference() {
    if (impl_->buffer) ggml_backend_buffer_free(impl_->buffer);
    if (impl_->compute_ctx) ggml_free(impl_->compute_ctx);
    if (impl_->backend) ggml_backend_free(impl_->backend);
}

bool GGMLInference::init(const GGUFModel& gguf) {
    auto& m = *impl_;
    m.meta = gguf.meta;
    m.n_layer = m.meta.n_layer;
    m.n_embd = m.meta.n_embd;
    m.n_head = m.meta.n_head;
    m.n_head_kv = m.meta.n_head_kv > 0 ? m.meta.n_head_kv : m.meta.n_head;
    m.n_ff = m.meta.n_ff;
    m.n_vocab = m.meta.n_vocab;
    m.head_dim = m.n_embd / m.n_head;
    
    std::cerr << "edge-llama: initializing ggml CPU backend..." << std::endl;
    
    // Init ggml backend
    ggml_backend_init_best();
    
    // Find CPU backend
    m.backend = ggml_backend_dev_init(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), nullptr);
    if (!m.backend) {
        std::cerr << "edge-llama: failed to init CPU backend" << std::endl;
        return false;
    }
    m.buft = ggml_backend_get_default_buffer_type(m.backend);
    
    // Calculate total weight size
    size_t total_size = 0;
    for (auto& [name, tensor] : gguf.tensors) {
        total_size += tensor->data.size();
    }
    
    std::cerr << "edge-llama: " << gguf.tensors.size() << " tensors, "
              << (total_size / 1024 / 1024) << " MB raw data" << std::endl;
    
    m.weight_data.resize(total_size);
    
    // Create weight context with precise sizing
    size_t ctx_size = 0;
    ctx_size += gguf.tensors.size() * (sizeof(ggml_tensor) + 128); // tensor objects
    ctx_size += total_size; // data
    ctx_size += 256 * 1024; // fudge
    
    auto* weight_ctx = ggml_init({ctx_size, nullptr, false});
    if (!weight_ctx) {
        std::cerr << "edge-llama: failed to init weight context" << std::endl;
        return false;
    }
    
    // Create and populate weight tensors
    m.w.layers.resize(m.n_layer);
    
    // Load tensor_embd
    m.load_tensor(weight_ctx, gguf, "token_embd.weight", &m.w.token_embd);
    m.load_tensor(weight_ctx, gguf, "output_norm.weight", &m.w.output_norm);
    m.load_tensor(weight_ctx, gguf, "output.weight", &m.w.output);
    
    for (int i = 0; i < m.n_layer; i++) {
        auto& l = m.w.layers[i];
        auto name = [&](const std::string& s) { 
            return "blk." + std::to_string(i) + "." + s; 
        };
        
        m.load_tensor(weight_ctx, gguf, name("attn_norm.weight"), &l.attn_norm);
        m.load_tensor(weight_ctx, gguf, name("attn_q.weight"), &l.attn_q);
        m.load_tensor(weight_ctx, gguf, name("attn_k.weight"), &l.attn_k);
        m.load_tensor(weight_ctx, gguf, name("attn_v.weight"), &l.attn_v);
        m.load_tensor(weight_ctx, gguf, name("attn_output.weight"), &l.attn_o);
        m.load_tensor(weight_ctx, gguf, name("ffn_norm.weight"), &l.ffn_norm);
        m.load_tensor(weight_ctx, gguf, name("ffn_gate.weight"), &l.ffn_gate);
        m.load_tensor(weight_ctx, gguf, name("ffn_down.weight"), &l.ffn_down);
        m.load_tensor(weight_ctx, gguf, name("ffn_up.weight"), &l.ffn_up);
        
        // Bias (optional)
        m.load_tensor(weight_ctx, gguf, name("attn_q.bias"), &l.attn_q_b);
        m.load_tensor(weight_ctx, gguf, name("attn_k.bias"), &l.attn_k_b);
        m.load_tensor(weight_ctx, gguf, name("attn_v.bias"), &l.attn_v_b);
    }
    
    // Allocate backend buffer and copy weights
    size_t weight_buf_size = ggml_backend_buffer_type_alloc_size(m.buft, weight_ctx);
    m.buffer = ggml_backend_buffer_alloc(m.buft, weight_buf_size);
    if (!m.buffer) {
        std::cerr << "edge-llama: failed to allocate weight buffer" << std::endl;
        ggml_free(weight_ctx);
        return false;
    }
    
    // Copy weight data to backend
    // For CPU backend, ggml_backend_tensor_copy copies data into the buffer
    for (auto& [name, tensor] : gguf.tensors) {
        // Find matching ggml tensor
        // We need to map from name to the ggml_tensor*
        // This is hacky but works for CPU backend
    }
    
    // Actually for CPU backend, the data is passed through
    // ggml_backend_buffer_alloc and tensor->data pointers are set up
    ggml_backend_buffer_copy_tensors(nullptr, m.buffer, weight_ctx);
    
    std::cerr << "edge-llama: ggml CPU backend ready" << std::endl;
    
    // Allocate KV cache
    int ctx_size_kv = 2048;
    m.kv.k.resize(m.n_layer * ctx_size_kv * m.n_head_kv * m.head_dim);
    m.kv.v.resize(m.n_layer * ctx_size_kv * m.n_head_kv * m.head_dim);
    m.kv.seq_len = 0;
    
    return true;
}

std::string GGMLInference::generate(const std::string& prompt, int max_tokens) {
    auto& m = *impl_;
    m.gen_start = std::chrono::steady_clock::now();
    m.kv.seq_len = 0;
    
    auto tokens = m.tokenizer.encode(prompt);
    std::cerr << "edge-llama: " << tokens.size() << " prompt tokens" << std::endl;
    
    // Build compute graph for a single forward pass
    // This uses ggml's ops: mul_mat, rms_norm, silu, rope, soft_max
    
    // Since building the full transformer graph is complex,
    // let's start with a single token test
    std::vector<int32_t> input;
    input.push_back(m.tokenizer.bos_token());
    input.insert(input.end(), tokens.begin(), tokens.end());
    
    std::string result;
    
    for (int step = 0; step < max_tokens && step < 5; step++) {
        // Create compute graph for this step
        auto* graph_ctx = ggml_init({8 * 1024 * 1024, nullptr, false});
        if (!graph_ctx) break;
        
        // Build: embed → transformer layers → output norm → lm_head
        // This is a simplified forward pass
        // Real implementation would use ggml_mul_mat etc.
        
        // For now, just do embedding lookup manually to verify the graph works
        int token_id = step == 0 ? input[0] : input.back();
        
        // Token embedding lookup using ggml_view + ggml_get_rows
        auto* emb = ggml_get_rows(graph_ctx, m.w.token_embd, 
            ggml_new_i32(graph_ctx, token_id));
        
        if (!emb) {
            ggml_free(graph_ctx);
            break;
        }
        
        // For debugging: just return the token embedding
        result += m.tokenizer.decode(token_id);
        
        // In the real version:
        // 1. RMS Norm → QKV
        // 2. RoPE + Attention
        // 3. Output projection
        // 4. FFN (gate+up+silu+down)
        // 5. Repeat for all layers
        // 6. Final norm + lm_head
        // 7. Sample
        
        ggml_free(graph_ctx);
    }
    
    auto end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(end - m.gen_start).count();
    m.stats.tokens_per_second = elapsed > 0 ? static_cast<int>(5 / elapsed) : 0;
    m.stats.total_tokens = 5;
    
    std::cerr << "edge-llama: test generated " << result.length() << " chars" << std::endl;
    
    return result;
}

GGMLInference::Stats GGMLInference::get_stats() const {
    return impl_->stats;
}

} // namespace edge_llama
