#ifndef EDGE_LLAMA_ENGINE_H
#define EDGE_LLAMA_ENGINE_H

#include <string>
#include <functional>
#include <memory>

struct llama_model;
struct llama_context;

namespace edge_llama {

struct EngineConfig {
    std::string model_path;
    int n_threads = 4;
    int n_ctx = 2048;
    int n_gpu_layers = 0;      // 0 = CPU only, >0 = offload layers to GPU
    float temperature = 0.7f;
    int n_predict = 256;
};

struct TokenStream {
    std::string text;
    bool done = false;
};

class Engine {
public:
    Engine();
    ~Engine();

    bool load(const EngineConfig& config);
    bool is_loaded() const;

    // Synchronous inference — returns complete response
    std::string generate(const std::string& prompt);

    // Streaming inference — callback per token
    void generate_stream(const std::string& prompt,
                         std::function<bool(const TokenStream&)> callback);

    // Performance stats
    struct Stats {
        int tokens_per_second = 0;
        int total_tokens = 0;
        int prompt_tokens = 0;
        int gpu_layers = 0;
        double load_time_ms = 0;
    };
    Stats get_stats() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace edge_llama

#endif // EDGE_LLAMA_ENGINE_H
