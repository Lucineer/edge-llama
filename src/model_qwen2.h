#ifndef EDGE_LLAMA_MODEL_QWEN2_H
#define EDGE_LLAMA_MODEL_QWEN2_H

#include "gguf_loader.h"
#include <vector>
#include <cstdint>

namespace edge_llama {

// Qwen2 model inference engine
// Loads tensors from GGUFModel, runs forward pass entirely on CPU
// Uses ggml compute graph for matrix operations

class Qwen2Model {
public:
    Qwen2Model();
    ~Qwen2Model();
    
    // Initialize from a loaded GGUF model
    bool init(const GGUFModel& model);
    
    // Forward pass: token IDs → logits
    // Returns n_vocab logits for the last token
    std::vector<float> forward(const std::vector<int32_t>& tokens);
    
    // Generate text from a prompt (tokenize + forward + sample)
    std::string generate(const std::string& prompt, int max_tokens = 256);
    
    // Tokenizer helpers
    std::vector<int32_t> tokenize(const std::string& text);
    std::string detokenize(int32_t token);
    
    // Statistics
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

#endif // EDGE_LLAMA_MODEL_QWEN2_H
