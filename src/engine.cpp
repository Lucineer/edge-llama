#include "engine.h"
#include <iostream>
#include <sstream>
#include <chrono>
#include <vector>

// llama.cpp headers from pip package
#include "llama.h"
#include "ggml.h"

namespace edge_llama {

struct Engine::Impl {
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    const llama_vocab* vocab = nullptr;
    EngineConfig config;
    Stats stats;
    std::chrono::time_point<std::chrono::steady_clock> load_start;
};

Engine::Engine() : impl_(std::make_unique<Impl>()) {}

Engine::~Engine() {
    if (impl_->ctx) llama_free(impl_->ctx);
    if (impl_->model) llama_model_free(impl_->model);
    llama_backend_free();
}

bool Engine::load(const EngineConfig& config) {
    impl_->load_start = std::chrono::steady_clock::now();
    impl_->config = config;

    // Initialize llama backend
    llama_backend_init();

    // Model parameters
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = config.n_gpu_layers;
    model_params.use_mmap = true;
    model_params.use_mlock = false;

    // Load model
    impl_->model = llama_model_load_from_file(config.model_path.c_str(), model_params);
    if (!impl_->model) {
        std::cerr << "edge-llama: failed to load model from " << config.model_path << std::endl;
        return false;
    }

    // Get vocabulary
    impl_->vocab = llama_model_get_vocab(impl_->model);
    if (!impl_->vocab) {
        std::cerr << "edge-llama: failed to get vocabulary" << std::endl;
        llama_model_free(impl_->model);
        impl_->model = nullptr;
        return false;
    }

    // Context parameters
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = config.n_ctx;
    ctx_params.n_threads = config.n_threads;
    ctx_params.n_threads_batch = config.n_threads;
    ctx_params.offload_kqv = (config.n_gpu_layers > 0);

    // Create context
    impl_->ctx = llama_init_from_model(impl_->model, ctx_params);
    if (!impl_->ctx) {
        std::cerr << "edge-llama: failed to create context" << std::endl;
        llama_model_free(impl_->model);
        impl_->model = nullptr;
        return false;
    }

    // Record stats
    auto load_end = std::chrono::steady_clock::now();
    impl_->stats.load_time_ms = std::chrono::duration<double, std::milli>(load_end - impl_->load_start).count();
    impl_->stats.gpu_layers = config.n_gpu_layers;

    std::cerr << "edge-llama: model loaded in " << impl_->stats.load_time_ms << "ms" << std::endl;
    return true;
}

bool Engine::is_loaded() const {
    return impl_->model != nullptr && impl_->ctx != nullptr;
}

std::string Engine::generate(const std::string& prompt) {
    if (!is_loaded()) return "";

    std::ostringstream result;
    generate_stream(prompt, [&result](const TokenStream& ts) {
        if (!ts.done) {
            result << ts.text;
        }
        return true; // continue
    });
    return result.str();
}

void Engine::generate_stream(const std::string& prompt,
                              std::function<bool(const TokenStream&)> callback) {
    if (!is_loaded()) return;

    auto start = std::chrono::steady_clock::now();
    int n_tokens = 0;

    // Tokenize the prompt
    int n_prompt_tokens = prompt.length() + 4;
    std::vector<llama_token> prompt_tokens(n_prompt_tokens);
    n_prompt_tokens = llama_tokenize(
        impl_->vocab,
        prompt.c_str(),
        prompt.length(),
        prompt_tokens.data(),
        n_prompt_tokens,
        true,   // add_bos
        false   // special
    );
    if (n_prompt_tokens < 0) {
        std::cerr << "edge-llama: tokenization failed" << std::endl;
        return;
    }
    prompt_tokens.resize(n_prompt_tokens);

    int n_predict = impl_->config.n_predict;
    std::vector<llama_token> tokens;

    // Process prompt
    {
        auto batch = llama_batch_get_one(prompt_tokens.data(), n_prompt_tokens);
        if (llama_decode(impl_->ctx, batch) != 0) {
            std::cerr << "edge-llama: prompt decode failed" << std::endl;
            return;
        }
    }

    // Main generation loop
    for (int i = 0; i < n_predict; i++) {
        // Sample the next token (greedy)
        auto* logits = llama_get_logits_ith(impl_->ctx, -1);
        int n_vocab = llama_vocab_n_tokens(impl_->vocab);

        llama_token new_token_id = 0;
        float max_logit = -1e10f;
        for (int j = 0; j < n_vocab; j++) {
            if (logits[j] > max_logit) {
                max_logit = logits[j];
                new_token_id = j;
            }
        }

        // Check for end of generation
        if (llama_vocab_is_eog(impl_->vocab, new_token_id)) {
            break;
        }

        // Convert token to text
        char buf[128];
        int n = llama_token_to_piece(impl_->vocab, new_token_id, buf, sizeof(buf), 0, true);
        if (n > 0) {
            std::string token_text(buf, n);
            n_tokens++;

            TokenStream ts;
            ts.text = token_text;
            ts.done = false;
            if (!callback(ts)) {
                break;
            }
        }

        // Prepare next batch
        tokens.push_back(new_token_id);
        auto batch = llama_batch_get_one(tokens.data(), tokens.size());
        if (llama_decode(impl_->ctx, batch) != 0) {
            std::cerr << "edge-llama: decode failed at step " << i << std::endl;
            break;
        }

        // For efficiency with greedy sampling, only keep last token
        tokens.clear();
    }

    // Signal done
    TokenStream done_ts;
    done_ts.done = true;
    callback(done_ts);

    // Record stats
    auto end = std::chrono::steady_clock::now();
    double elapsed_s = std::chrono::duration<double>(end - start).count();
    impl_->stats.tokens_per_second = (elapsed_s > 0) ? static_cast<int>(n_tokens / elapsed_s) : 0;
    impl_->stats.total_tokens = n_tokens;
    impl_->stats.prompt_tokens = n_prompt_tokens;
}

Engine::Stats Engine::get_stats() const {
    return impl_->stats;
}

} // namespace edge_llama
