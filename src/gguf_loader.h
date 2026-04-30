#ifndef EDGE_LLAMA_GGUF_LOADER_H
#define EDGE_LLAMA_GGUF_LOADER_H

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <memory>

namespace edge_llama {

// GGUF format constants
constexpr uint32_t GGUF_MAGIC = 0x46554747; // "GGUF"
constexpr uint32_t GGUF_VERSION = 3;

// GGML tensor types (from ggml.h)
enum class GGMLType : uint32_t {
    F32     = 0,
    F16     = 1,
    Q4_0    = 2,
    Q4_1    = 3,
    Q5_0    = 6,
    Q5_1    = 7,
    Q8_0    = 8,
    Q8_1    = 9,
    Q2_K    = 10,
    Q3_K    = 11,
    Q4_K    = 12,
    Q5_K    = 13,
    Q6_K    = 14,
    Q8_K    = 15,
    IQ2_XXS = 16,
    IQ2_XS  = 17,
    IQ3_XXS = 18,
    IQ1_S   = 19,
    IQ4_NL  = 20,
    IQ3_S   = 21,
    IQ2_S   = 22,
    IQ4_XS  = 23,
    I8      = 24,
    I16     = 25,
    I32     = 26,
};

// A loaded tensor
struct Tensor {
    std::string name;
    GGMLType type;
    std::vector<int64_t> dims;
    int64_t n_elements() const;
    
    // Raw weight data
    std::vector<uint8_t> data;
};

// Model hyperparameters from GGUF metadata
struct ModelMeta {
    std::string arch;          // "qwen2", "llama", etc.
    std::string name;
    uint32_t n_layer = 0;
    uint32_t n_embd = 0;
    uint32_t n_head = 0;
    uint32_t n_head_kv = 0;
    uint32_t n_ff = 0;
    uint32_t n_vocab = 0;
    uint32_t n_expert = 0;
    float f_norm_rms_eps = 1e-6f;
    float freq_base = 10000.0f;
};

// Loaded model state (tensor metadata + raw data)
struct GGUFModel {
    ModelMeta meta;
    std::unordered_map<std::string, std::unique_ptr<Tensor>> tensors;
    
    const Tensor* get(const std::string& name) const;
    Tensor* get_mut(const std::string& name);
};

// Load a GGUF file
bool load_gguf(const std::string& path, GGUFModel& model);

// Print metadata
void print_meta(const GGUFModel& model);

} // namespace edge_llama

#endif // EDGE_LLAMA_GGUF_LOADER_H
