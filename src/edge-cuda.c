// edge-cuda.c — The Metal API Implementation
//
// Links ggml-base, ggml-cpu, ggml-cuda libraries
// and provides a clean C API for inference.
//
// Build:
//   gcc -shared -fPIC -o libedge-cuda.so.0.5 \
//       edge-cuda.c gguf_loader.cpp \
//       -I/path/to/ggml/include \
//       -L/path/to/ggml/lib -lggml -lggml-cpu -lggml-cuda -lggml-base \
//       -lpthread -ldl -lstdc++ -lm

#include "edge-cuda.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

// Forward-decl the C++ implementation
struct edge_impl;

// ── Thread-local error buffer ──
static __thread char edge_error[256] = {0};

const char* edge_last_error(void) {
    return edge_error;
}

// ── Internal: load via C++ ggml backend ──
// We use a C++ compilation unit that #includes ggml headers.
// This C file declares the interface.

// These are implemented in edge-cuda-impl.cpp
struct edge_impl* edge_impl_load(const char* path);
void edge_impl_unload(struct edge_impl* impl);
float* edge_impl_forward(struct edge_impl* impl, const int32_t* tokens, int32_t n);
char* edge_impl_generate(struct edge_impl* impl, const char* prompt, int32_t max_tokens, int32_t* out_len, int32_t* new_tokens);
char* edge_impl_generate_stream(struct edge_impl* impl, const char* prompt, int32_t max_tokens, int32_t* out_len, int32_t* new_tokens, edge_stream_cb callback, void* user_ctx);
int32_t edge_impl_n_layer(struct edge_impl* impl);
int32_t edge_impl_n_embd(struct edge_impl* impl);
int32_t edge_impl_n_head(struct edge_impl* impl);
int32_t edge_impl_n_vocab(struct edge_impl* impl);
const char* edge_impl_backend(struct edge_impl* impl);
int64_t edge_impl_vram_total(struct edge_impl* impl);
int64_t edge_impl_vram_free(struct edge_impl* impl);
int32_t edge_impl_tps(struct edge_impl* impl);

// ── Public API Implementations ──

struct edge_ctx {
    struct edge_impl* impl;
    void* dl_handle; // Only used if loaded via dlopen
};

edge_t* edge_load(const char* gguf_path) {
    edge_t* ctx = (edge_t*)calloc(1, sizeof(edge_t));
    if (!ctx) return NULL;
    
    ctx->impl = edge_impl_load(gguf_path);
    if (!ctx->impl) {
        snprintf(edge_error, sizeof(edge_error), "Failed to load: %s", gguf_path);
        free(ctx);
        return NULL;
    }
    
    return ctx;
}

void edge_unload(edge_t* ctx) {
    if (!ctx) return;
    edge_impl_unload(ctx->impl);
    if (ctx->dl_handle) {
        // dlclose handled elsewhere
    }
    free(ctx);
}

float* edge_forward(edge_t* ctx, const int32_t* tokens, int32_t n_tokens) {
    return edge_impl_forward(ctx->impl, tokens, n_tokens);
}

char* edge_generate(edge_t* ctx, const char* prompt,
                    int32_t max_tokens, int32_t* out_len,
                    int32_t* new_tokens) {
    return edge_impl_generate(ctx->impl, prompt, max_tokens, out_len, new_tokens);
}

char* edge_generate_stream(edge_t* ctx, const char* prompt,
                           int32_t max_tokens, int32_t* out_len,
                           int32_t* new_tokens,
                           edge_stream_cb callback, void* user_ctx) {
    return edge_impl_generate_stream(ctx->impl, prompt, max_tokens, out_len, new_tokens, callback, user_ctx);
}

int32_t edge_n_layer(edge_t* ctx) { return edge_impl_n_layer(ctx->impl); }
int32_t edge_n_embd(edge_t* ctx) { return edge_impl_n_embd(ctx->impl); }
int32_t edge_n_head(edge_t* ctx) { return edge_impl_n_head(ctx->impl); }
int32_t edge_n_vocab(edge_t* ctx) { return edge_impl_n_vocab(ctx->impl); }
const char* edge_backend(edge_t* ctx) { return edge_impl_backend(ctx->impl); }
int64_t edge_vram_total(edge_t* ctx) { return edge_impl_vram_total(ctx->impl); }
int64_t edge_vram_free(edge_t* ctx) { return edge_impl_vram_free(ctx->impl); }
int32_t edge_tokens_per_second(edge_t* ctx) { return edge_impl_tps(ctx->impl); }

void edge_free_string(char* s) {
    free(s);
}

// ── Test main (compiled when not building as shared lib) ──
#ifndef EDGE_AS_SHARED_LIB
int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s model.gguf [prompt]\n", argv[0]);
        return 1;
    }
    
    printf("edge-cuda: loading %s...\n", argv[1]);
    edge_t* e = edge_load(argv[1]);
    if (!e) {
        fprintf(stderr, "edge-cuda: %s\n", edge_last_error());
        return 1;
    }
    
    printf("  backend:   %s\n", edge_backend(e));
    printf("  layers:    %d\n", edge_n_layer(e));
    printf("  heads:     %d\n", edge_n_head(e));
    printf("  embd:      %d\n", edge_n_embd(e));
    printf("  vocab:     %d\n", edge_n_vocab(e));
    printf("  vram:      %ld MB / %ld MB\n",
           edge_vram_free(e) / 1048576,
           edge_vram_total(e) / 1048576);
    
    const char* prompt = argc >= 3 ? argv[2] : "Hello, AI. ";
    printf("\n  prompt:    \"%s\"\n", prompt);
    
    int32_t out_len, new_tokens;
    char* result = edge_generate(e, prompt, 64, &out_len, &new_tokens);
    
    printf("  result:    \"%s\"\n", result ? result : "(null)");
    printf("  generated: %d tokens\n", new_tokens);
    printf("  speed:     %d t/s\n", edge_tokens_per_second(e));
    
    edge_free_string(result);
    edge_unload(e);
    return 0;
}
#endif
