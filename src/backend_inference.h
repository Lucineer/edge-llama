#ifndef EDGE_LLAMA_BACKEND_INFERENCE_H
#define EDGE_LLAMA_BACKEND_INFERENCE_H

#include "gguf_loader.h"
#include <string>
#include <vector>
#include <memory>

namespace edge_llama {

/// Full inference engine using ggml backends.
/// Detects CPU/CUDA at runtime. Uses CUDA when available, CPU fallback otherwise.
class BackendInference {
public:
    BackendInference();
    ~BackendInference();
    
    /// Initialize with GGUF weights and select best available backend
    bool init(const GGUFModel& gguf);
    
    /// Full generate: tokenize, forward, sample
    std::string generate(const std::string& prompt, int max_tokens = 256);
    
    struct Stats {
        int tokens_per_second = 0;
        int total_tokens = 0;
        int prompt_tokens = 0;
        std::string backend_name = "none";
        int64_t vram_total_mb = 0;
        int64_t vram_used_mb = 0;
    };
    Stats get_stats() const;
    
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    void log_stats();
};

} // namespace edge_llama

#endif // EDGE_LLAMA_BACKEND_INFERENCE_H
