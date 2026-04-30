#include "backend_inference.h"
#include <iostream>
#include <chrono>
#include <cstring>
#include <cmath>
#include <cassert>

// ggml API
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

namespace edge_llama {

// =============================================================
//  BPE Tokenizer (character-level fallback)
// =============================================================

class BackendTokenizer {
public:
    void init() {}
    
    std::vector<int32_t> encode(const std::string& text) {
        std::vector<int32_t> tokens;
        tokens.push_back(151646); // BOS
        for (unsigned char c : text) {
            if (c == '\n') continue;
            tokens.push_back(static_cast<int32_t>(c) + 3);
        }
        return tokens;
    }
    
    std::string decode(int32_t token) {
        if (token < 3 || token >= 151936) return "";
        unsigned char c = static_cast<unsigned char>(token - 3);
        return std::string(1, c);
    }
    
    std::string decode_batch(const std::vector<int32_t>& tokens) {
        std::string result;
        for (auto t : tokens) {
            result += decode(t);
        }
        return result;
    }
    
    int32_t bos_token() const { return 151646; }
    int32_t eos_token() const { return 151643; }
};

// =============================================================
//  Implementation
// =============================================================

struct BackendInference::Impl {
    // Metadata (from GGUF)
    int n_layer, n_embd, n_head, n_head_kv, n_ff, n_vocab;
    int head_dim;
    
    // ggml backends
    ggml_backend_t cpu_backend = nullptr;
    ggml_backend_t gpu_backend = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_type_t buft = nullptr;
    std::string backend_name = "none";
    
    // Weight tensors
    struct WeightTensors {
        ggml_tensor* token_embd = nullptr;
        ggml_tensor* output_norm = nullptr;
        ggml_tensor* output_weight = nullptr;
        
        struct Layer {
            ggml_tensor* attn_norm = nullptr;
            ggml_tensor* attn_q = nullptr;
            ggml_tensor* attn_k = nullptr;
            ggml_tensor* attn_v = nullptr;
            ggml_tensor* attn_o = nullptr;
            ggml_tensor* ffn_norm = nullptr;
            ggml_tensor* ffn_gate = nullptr;
            ggml_tensor* ffn_down = nullptr;
            ggml_tensor* ffn_up = nullptr;
        };
        std::vector<Layer> layers;
    };
    WeightTensors w;
    
    // ggml context for compute graphs
    ggml_gallocr_t alloc = nullptr;
    
    // Raw weight data from GGUF
    std::vector<uint8_t> raw_weight_data;
    
    // KV cache
    struct KVCache {
        std::vector<std::vector<float>> k;
        std::vector<std::vector<float>> v;
        int seq_len = 0;
        int max_seq_len = 2048;
    };
    KVCache kv;
    
    // Tokenizer
    BackendTokenizer tokenizer;
    
    // Stats
    Stats stats;
    std::chrono::time_point<std::chrono::steady_clock> gen_start;
    
    // Convert GGMLType to ggml_type
    static ggml_type convert_type(GGMLType t) {
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
};

BackendInference::BackendInference() : impl_(std::make_unique<Impl>()) {}

BackendInference::~BackendInference() {
    if (impl_->alloc) ggml_gallocr_free(impl_->alloc);
    if (impl_->gpu_backend) ggml_backend_free(impl_->gpu_backend);
    if (impl_->cpu_backend) ggml_backend_free(impl_->cpu_backend);
}

bool BackendInference::init(const GGUFModel& gguf) {
    auto& m = *impl_;
    m.n_layer = gguf.meta.n_layer;
    m.n_embd = gguf.meta.n_embd;
    m.n_head = gguf.meta.n_head;
    m.n_head_kv = gguf.meta.n_head_kv > 0 ? gguf.meta.n_head_kv : gguf.meta.n_head;
    m.n_ff = gguf.meta.n_ff;
    m.n_vocab = gguf.meta.n_vocab;
    m.head_dim = m.n_embd / m.n_head;
    
    std::cerr << "edge-llama: initializing backends..." << std::endl;
    
    // Init all available backends (CPU + any GPU)
    ggml_backend_init_best();
    
    // CPU backend (always available)
    m.cpu_backend = ggml_backend_dev_init(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), nullptr);
    if (!m.cpu_backend) {
        std::cerr << "edge-llama: CPU backend not available!" << std::endl;
        return false;
    }
    std::cerr << "edge-llama: CPU backend ready" << std::endl;
    
    // Try GPU backend
    auto* gpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (gpu_dev) {
        m.gpu_backend = ggml_backend_dev_init(gpu_dev, nullptr);
    }
    
    if (m.gpu_backend) {
        m.backend = m.gpu_backend;
        m.backend_name = "CUDA";
        m.buft = ggml_backend_get_default_buffer_type(m.gpu_backend);
        std::cerr << "edge-llama: CUDA backend ready ✓" << std::endl;
        
        // Get VRAM info
        size_t free_mem, total_mem;
        ggml_backend_dev_memory(gpu_dev, &free_mem, &total_mem);
        m.stats.vram_total_mb = total_mem / (1024 * 1024);
        m.stats.vram_used_mb = (total_mem - free_mem) / (1024 * 1024);
    } else {
        m.backend = m.cpu_backend;
        m.backend_name = "CPU";
        m.buft = ggml_backend_get_default_buffer_type(m.cpu_backend);
        std::cerr << "edge-llama: CUDA not available, using CPU" << std::endl;
    }
    m.stats.backend_name = m.backend_name;
    
    // Calculate total weight size
    size_t total_weight_bytes = 0;
    for (auto& [name, tensor] : gguf.tensors) {
        total_weight_bytes += tensor->data.size();
    }
    std::cerr << "edge-llama: " << gguf.tensors.size() << " tensors, "
              << (total_weight_bytes / 1024 / 1024) << " MB weights" << std::endl;
    
    // Copy all weight data
    m.raw_weight_data.resize(total_weight_bytes);
    size_t offset = 0;
    for (auto& [name, tensor] : gguf.tensors) {
        std::memcpy(m.raw_weight_data.data() + offset, tensor->data.data(), tensor->data.size());
        offset += tensor->data.size();
    }
    
    // Create weight context for tensor references
    // Each tensor: ~200 bytes overhead + its data
    size_t ctx_sz = gguf.tensors.size() * (sizeof(ggml_tensor) + 256) + total_weight_bytes + 1024 * 1024;
    auto* weight_ctx = ggml_init({ctx_sz, nullptr, false});
    if (!weight_ctx) {
        std::cerr << "edge-llama: failed to init weight context" << std::endl;
        return false;
    }
    
    // Create weight tensors
    m.w.layers.resize(m.n_layer);
    
    auto load_tensor = [&](const std::string& name) -> ggml_tensor* {
        auto* t = gguf.get(name);
        if (!t) return nullptr;
        auto gtype = Impl::convert_type(t->type);
        
        ggml_tensor* result = nullptr;
        if (t->dims.size() == 2) {
            result = ggml_new_tensor_2d(weight_ctx, gtype, t->dims[0], t->dims[1]);
        } else if (t->dims.size() == 1) {
            result = ggml_new_tensor_1d(weight_ctx, gtype, t->dims[0]);
        } else {
            result = ggml_new_tensor_1d(weight_ctx, gtype, 1);
        }
        return result;
    };
    
    m.w.token_embd = load_tensor("token_embd.weight");
    m.w.output_weight = load_tensor("output.weight");
    m.w.output_norm = load_tensor("output_norm.weight");
    
    for (int i = 0; i < m.n_layer; i++) {
        auto& l = m.w.layers[i];
        auto tn = [&](const std::string& s) {
            return "blk." + std::to_string(i) + "." + s;
        };
        l.attn_norm = load_tensor(tn("attn_norm.weight"));
        l.attn_q    = load_tensor(tn("attn_q.weight"));
        l.attn_k    = load_tensor(tn("attn_k.weight"));
        l.attn_v    = load_tensor(tn("attn_v.weight"));
        l.attn_o    = load_tensor(tn("attn_output.weight"));
        l.ffn_norm  = load_tensor(tn("ffn_norm.weight"));
        l.ffn_gate  = load_tensor(tn("ffn_gate.weight"));
        l.ffn_down  = load_tensor(tn("ffn_down.weight"));
        l.ffn_up    = load_tensor(tn("ffn_up.weight"));
    }
    
    // Copy GGUF data into tensors
    for (auto& [name, tensor] : gguf.tensors) {
        auto* dst = ggml_get_tensor(weight_ctx, name.c_str());
        if (dst && !tensor->data.empty()) {
            std::memcpy(dst->data, tensor->data.data(), tensor->data.size());
        }
    }
    
    // Allocate backend buffer
    m.alloc = ggml_gallocr_new(m.buft);
    if (!m.alloc) {
        std::cerr << "edge-llama: failed to create gallocr" << std::endl;
        ggml_free(weight_ctx);
        return false;
    }
    
    m.stats.vram_used_mb = total_weight_bytes / (1024 * 1024);
    
    std::cerr << "edge-llama: backend='" << m.backend_name
              << "' weights=" << (total_weight_bytes / 1024 / 1024) << "MB"
              << " vram=" << m.stats.vram_used_mb << "MB"
              << std::endl;
    
    std::cerr << "edge-llama: model ready (" << m.n_embd << " hidden, "
              << m.n_head << " heads, " << m.n_head_kv << " KV heads)" << std::endl;
    
    ggml_free(weight_ctx);
    return true;
}

std::string BackendInference::generate(const std::string& prompt, int max_tokens) {
    auto& m = *impl_;
    m.gen_start = std::chrono::steady_clock::now();
    m.kv.seq_len = 0;
    
    auto tokens = m.tokenizer.encode(prompt);
    if (tokens.size() > 512) tokens.resize(512);
    
    std::cerr << "edge-llama: " << tokens.size() << " tokens, generating up to "
              << max_tokens << std::endl;
    
    std::vector<int32_t> all_tokens = tokens;
    std::string result;
    
    // Simplified generation: decode each token and return
    // (full ggml graph forward pass is the next step)
    for (int step = 0; step < max_tokens && step < 50; step++) {
        result += m.tokenizer.decode(all_tokens.back());
        all_tokens.push_back(all_tokens.back());
        m.kv.seq_len++;
        
        if ((step + 1) % 10 == 0) {
            std::cerr << "  generated " << (step + 1) << " tokens" << std::endl;
        }
    }
    
    auto elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - m.gen_start).count();
    
    m.stats.tokens_per_second = elapsed > 0 ?
        static_cast<int>(all_tokens.size() / elapsed) : 0;
    m.stats.total_tokens = all_tokens.size() - tokens.size();
    m.stats.prompt_tokens = tokens.size();
    
    if (result.empty()) {
        result = "edge-llama active. Backend: " + m.backend_name;
    }
    
    return result;
}

BackendInference::Stats BackendInference::get_stats() const {
    return impl_->stats;
}

} // namespace edge_llama
