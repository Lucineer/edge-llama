#ifndef EDGE_LLAMA_GGML_INFERENCE_H
#define EDGE_LLAMA_GGML_INFERENCE_H

#include "gguf_loader.h"
#include <string>
#include <vector>
#include <memory>

// Forward declare ggml types
struct ggml_context;
struct ggml_backend;
struct ggml_backend_buffer_type;

namespace edge_llama {

// Qwen2 model inference using ggml compute graphs
class GGMLInference {
public:
    GGMLInference();
    ~GGMLInference();
    
    // Initialize from GGUF model weights
    bool init(const GGUFModel& gguf);
    
    // Generate text from prompt
    std::string generate(const std::string& prompt, int max_tokens = 128);
    
    struct Stats {
        int tokens_per_second = 0;
        int total_tokens = 0;
        int prompt_tokens = 0;
    };
    Stats get_stats() const;
    
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace edge_llama

#endif // EDGE_LLAMA_GGML_INFERENCE_H
