// edge-cuda.h — The Metal API
// 
// Single header providing inference through ggml backends.
// Load as shared library (.so) from ANY language.
// Zero-copy via shared memory. Auto-selects CPU/CUDA.
//
// Usage (C):
//   edge_t* e = edge_load("model.gguf");
//   float* out = edge_forward(e, tokens, n);
//   edge_unload(e);
//
// Usage (Python/ctypes):
//   lib = ctypes.CDLL("libedge-cuda.so.0")
//   lib.edge_load.restype = ctypes.c_void_p
//   e = lib.edge_load(b"model.gguf")

#ifndef EDGE_CUDA_H
#define EDGE_CUDA_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle (internal = BackendInference via ggml_cuda/CPU)
typedef struct edge_ctx edge_t;

// ── Lifecycle ──

// Load a GGUF model. Returns NULL on failure.
// Backend selection: tries CUDA first, falls back to CPU.
edge_t* edge_load(const char* gguf_path);

// Unload model and free all resources.
void edge_unload(edge_t* ctx);

// ── Inference ──

// Forward pass: process token IDs, produce logits.
// tokens: int32 array of token IDs
// n_tokens: number of tokens
// Returns pointer to logits array (n_vocab floats).
// Pointer is valid until next call.
float* edge_forward(edge_t* ctx, const int32_t* tokens, int32_t n_tokens);

// Generate: convenience function for single prompt → string.
// Returns allocated string (caller must edge_free() it).
// *out_len receives number of bytes.
// *new_tokens receives number of tokens generated.
char* edge_generate(edge_t* ctx, const char* prompt,
                    int32_t max_tokens, int32_t* out_len,
                    int32_t* new_tokens);

// ── Introspection ──

// Get model metadata
int32_t edge_n_layer(edge_t* ctx);
int32_t edge_n_embd(edge_t* ctx);
int32_t edge_n_head(edge_t* ctx);
int32_t edge_n_vocab(edge_t* ctx);

// Get backend info
const char* edge_backend(edge_t* ctx);     // "CUDA" or "CPU"
int64_t edge_vram_total(edge_t* ctx);      // bytes
int64_t edge_vram_free(edge_t* ctx);       // bytes
int32_t edge_tokens_per_second(edge_t* ctx);

// ── Stream Callback ──

// Callback type: called for each generated piece during streaming generation.
// piece: pointer to UTF-8 text piece (may be partial token!)
// len:   byte length of this piece
// ctx:   user-provided context pointer (passed through from edge_generate_stream)
typedef void (*edge_stream_cb)(const char* piece, int32_t len, void* user_ctx);

// Generate with streaming callback.
// Same as edge_generate but calls cb(piece, len, user_ctx) for each generated piece.
// Still returns the full concatenated string (caller must edge_free_string).
char* edge_generate_stream(edge_t* ctx, const char* prompt,
                           int32_t max_tokens, int32_t* out_len,
                           int32_t* new_tokens,
                           edge_stream_cb callback, void* user_ctx);

// Free a string allocated by edge_generate
void edge_free_string(char* s);

// ── Shared Memory (zero-copy between processes) ──

// Share model weights via shm. Returns shm name.
// Other processes can edge_attach(name) to get same weights.
const char* edge_shm_export(edge_t* ctx);

// Attach to shared model exported by another process.
edge_t* edge_shm_attach(const char* shm_name);

// Detach from shared model (doesn't unload weights).
void edge_shm_detach(edge_t* ctx);

#ifdef __cplusplus
}
#endif

#endif // EDGE_CUDA_H
