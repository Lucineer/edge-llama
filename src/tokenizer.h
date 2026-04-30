#ifndef EDGE_LLAMA_TOKENIZER_H
#define EDGE_LLAMA_TOKENIZER_H

#include <string>
#include <vector>
#include <cstdint>

namespace edge_llama {

/// Character-level tokenizer (model-agnostic fallback)
/// Maps bytes to token IDs by adding 3 (BPE byte-level offset)
struct Tokenizer {
    int32_t bos = 151646;
    int32_t eos = 151643;
    
    std::vector<int32_t> encode(const std::string& text) const {
        std::vector<int32_t> tokens;
        tokens.push_back(bos);
        for (unsigned char c : text) {
            tokens.push_back(static_cast<int32_t>(c) + 3);
        }
        return tokens;
    }
    
    std::string decode(int32_t token) const {
        if (token < 3 || token >= 151936) return "";
        return std::string(1, static_cast<unsigned char>(token - 3));
    }
};

} // namespace edge_llama

#endif
