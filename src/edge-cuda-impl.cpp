// edge-cuda-impl.cpp — Implementation using llama.cpp v0.9.8+ API
//
// Links libllama.so directly. Uses modern llama_init_from_model, etc.
// Does NOT use deprecated functions.

#include "edge-cuda.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <string>
#include <mutex>

extern "C" {
#include "llama.h"
}

static thread_local char edge_error[256] = {0};

const char* edge_last_error(void) {
    return edge_error;
}

// ── Implementation ──

struct edge_impl {
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    llama_sampler* smpl = nullptr;
    const llama_vocab* vocab = nullptr;
    
    int32_t n_layer, n_embd, n_head, n_vocab;
    std::string backend_name = "llama.cpp (CPU)";
    double last_gen_time = 0.0;
    int last_gen_tokens = 0;
    std::mutex mtx;
};

extern "C" struct edge_impl* edge_impl_load(const char* path) {
    auto impl = new edge_impl();
    
    fprintf(stderr, "edge: loading model from %s\n", path);
    
    // Model params
    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = 99;  // Try full GPU offload
    
    impl->model = llama_model_load_from_file(path, mparams);
    if (!impl->model) {
        fprintf(stderr, "edge: retrying CPU-only\n");
        mparams.n_gpu_layers = 0;
        impl->model = llama_model_load_from_file(path, mparams);
        if (!impl->model) {
            snprintf(edge_error, sizeof(edge_error), "failed to load model: %s", path);
            delete impl;
            return nullptr;
        }
    }
    
    // Get metadata
    impl->n_layer = llama_model_n_layer(impl->model);
    impl->n_embd  = llama_model_n_embd(impl->model);
    impl->n_head  = llama_model_n_head(impl->model);
    impl->vocab   = llama_model_get_vocab(impl->model);
    impl->n_vocab = llama_vocab_n_tokens(impl->vocab);
    
    fprintf(stderr, "edge: %d layers, %d embd, %d heads, %d vocab\n",
            impl->n_layer, impl->n_embd, impl->n_head, impl->n_vocab);
    
    // Create context
    auto cparams = llama_context_default_params();
    cparams.n_ctx = 2048;
    cparams.n_batch = 512;
    cparams.n_threads = 4;
    cparams.n_threads_batch = 4;
    
    impl->ctx = llama_init_from_model(impl->model, cparams);
    if (!impl->ctx) {
        snprintf(edge_error, sizeof(edge_error), "failed to create context");
        llama_model_free(impl->model);
        delete impl;
        return nullptr;
    }
    
    // Greedy sampler
    impl->smpl = llama_sampler_init_greedy();
    
    fprintf(stderr, "edge: model ready\n");
    return impl;
}

extern "C" void edge_impl_unload(struct edge_impl* impl) {
    if (!impl) return;
    if (impl->smpl) llama_sampler_free(impl->smpl);
    if (impl->ctx) llama_free(impl->ctx);
    if (impl->model) llama_model_free(impl->model);
    delete impl;
}

extern "C" float* edge_impl_forward(struct edge_impl* impl, const int32_t* tokens, int32_t n) {
    return nullptr;
}

// ── Shared generate loop (used by both blocking and streaming) ──
typedef void (*edge_stream_cb)(const char* piece, int32_t len, void* user_ctx);

static std::string do_generate(edge_impl* impl, const char* prompt,
                               int32_t max_tokens, int32_t* out_n_gen,
                               edge_stream_cb callback, void* user_ctx) {
    auto t0 = std::chrono::steady_clock::now();
    
    // Tokenize
    int n_tokens = -llama_tokenize(impl->vocab, prompt, strlen(prompt), nullptr, 0, true, false);
    std::vector<llama_token> tokens(n_tokens);
    llama_tokenize(impl->vocab, prompt, strlen(prompt), tokens.data(), n_tokens, true, false);
    
    fprintf(stderr, "edge: %zu prompt tokens\n", tokens.size());
    
    std::string result;
    result.reserve(max_tokens * 16);
    
    // Batch for prompt
    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    if (llama_decode(impl->ctx, batch)) {
        fprintf(stderr, "edge: prompt decode failed\n");
        *out_n_gen = 0;
        return "";
    }
    
    // Generate
    int n_gen = 0;
    
    for (int i = 0; i < max_tokens; i++) {
        // Sample
        llama_token new_token = llama_sampler_sample(impl->smpl, impl->ctx, -1);
        
        if (llama_vocab_is_eog(impl->vocab, new_token)) {
            fprintf(stderr, "edge: EOS token\n");
            break;
        }
        
        // Decode token to printable piece
        char buf[64];
        int n = llama_token_to_piece(impl->vocab, new_token, buf, sizeof(buf), 0, false);
        if (n > 0) {
            result.append(buf, n);
            // Call stream callback if provided
            if (callback) {
                callback(buf, n, user_ctx);
            }
        }
        n_gen++;
        
        // Feed back
        llama_batch next = llama_batch_get_one(&new_token, 1);
        if (llama_decode(impl->ctx, next)) {
            fprintf(stderr, "edge: decode failed at step %d\n", i);
            break;
        }
    }
    
    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    impl->last_gen_time = elapsed;
    impl->last_gen_tokens = n_gen;
    
    *out_n_gen = n_gen;
    fprintf(stderr, "edge: %d tokens in %.2f s (%.0f t/s)\n", n_gen, elapsed, n_gen/elapsed);
    return result;
}

extern "C" char* edge_impl_generate(struct edge_impl* impl, const char* prompt,
                                    int32_t max_tokens, int32_t* out_len,
                                    int32_t* new_tokens) {
    std::lock_guard<std::mutex> lock(impl->mtx);
    int32_t n_gen = 0;
    std::string result = do_generate(impl, prompt, max_tokens, &n_gen, nullptr, nullptr);
    if (out_len) *out_len = (int32_t)result.size();
    if (new_tokens) *new_tokens = n_gen;
    if (result.empty()) return strdup("");
    char* c_result = (char*)malloc(result.size() + 1);
    if (c_result) {
        memcpy(c_result, result.c_str(), result.size());
        c_result[result.size()] = 0;
    }
    return c_result;
}

extern "C" char* edge_impl_generate_stream(struct edge_impl* impl, const char* prompt,
                                           int32_t max_tokens, int32_t* out_len,
                                           int32_t* new_tokens,
                                           edge_stream_cb callback, void* user_ctx) {
    std::lock_guard<std::mutex> lock(impl->mtx);
    int32_t n_gen = 0;
    std::string result = do_generate(impl, prompt, max_tokens, &n_gen, callback, user_ctx);
    if (out_len) *out_len = (int32_t)result.size();
    if (new_tokens) *new_tokens = n_gen;
    if (result.empty()) return strdup("");
    char* c_result = (char*)malloc(result.size() + 1);
    if (c_result) {
        memcpy(c_result, result.c_str(), result.size());
        c_result[result.size()] = 0;
    }
    return c_result;
}

extern "C" int32_t edge_impl_n_layer(struct edge_impl* impl) { return impl->n_layer; }
extern "C" int32_t edge_impl_n_embd(struct edge_impl* impl) { return impl->n_embd; }
extern "C" int32_t edge_impl_n_head(struct edge_impl* impl) { return impl->n_head; }
extern "C" int32_t edge_impl_n_vocab(struct edge_impl* impl) { return impl->n_vocab; }
extern "C" const char* edge_impl_backend(struct edge_impl* impl) { return impl->backend_name.c_str(); }
extern "C" int64_t edge_impl_vram_total(struct edge_impl* impl) { return 0; }
extern "C" int64_t edge_impl_vram_free(struct edge_impl* impl) { return 0; }
extern "C" int32_t edge_impl_tps(struct edge_impl* impl) {
    if (impl->last_gen_time <= 0) return 0;
    return (int32_t)(impl->last_gen_tokens / impl->last_gen_time);
}
