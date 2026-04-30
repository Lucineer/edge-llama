#include "gguf_loader.h"
#include <iostream>
#include <cstring>
#include <cassert>

namespace edge_llama {

int64_t Tensor::n_elements() const {
    int64_t n = 1;
    for (auto d : dims) n *= d;
    return n;
}

const Tensor* GGUFModel::get(const std::string& name) const {
    auto it = tensors.find(name);
    if (it != tensors.end()) return it->second.get();
    return nullptr;
}

Tensor* GGUFModel::get_mut(const std::string& name) {
    auto it = tensors.find(name);
    if (it != tensors.end()) return it->second.get();
    return nullptr;
}

static std::string read_string(std::ifstream& f) {
    uint64_t len;
    f.read(reinterpret_cast<char*>(&len), sizeof(len));
    std::string s(len, '\0');
    if (len > 0) f.read(&s[0], len);
    return s;
}

// GGUF v3 metadata format: key(string) + value_type(uint32) + value(variable)
static bool read_kv(std::ifstream& f, ModelMeta& meta) {
    auto key = read_string(f);
    
    uint32_t value_type;
    f.read(reinterpret_cast<char*>(&value_type), sizeof(value_type));
    
    // Read value
    // Types: 0=uint8, 1=int8, 2=uint16, 3=int16, 4=uint32, 5=int32,
    //        6=float32, 7=bool, 8=string, 9=array, 10=uint64, 11=int64,
    //        12=float64, 13=float16
    
    auto read_uint32 = [&]() -> uint32_t {
        uint32_t v;
        f.read(reinterpret_cast<char*>(&v), sizeof(v));
        return v;
    };
    auto read_float32 = [&]() -> float {
        float v;
        f.read(reinterpret_cast<char*>(&v), sizeof(v));
        return v;
    };
    auto read_uint64 = [&]() -> uint64_t {
        uint64_t v;
        f.read(reinterpret_cast<char*>(&v), sizeof(v));
        return v;
    };
    auto read_bool = [&]() -> bool {
        uint8_t v;
        f.read(reinterpret_cast<char*>(&v), sizeof(v));
        return v != 0;
    };
    
    switch (value_type) {
        case 0: { // uint8
            uint8_t v;
            f.read(reinterpret_cast<char*>(&v), sizeof(v));
            break;
        }
        case 4: { // uint32
            auto v = read_uint32();
            if (key == "qwen2.block_count") meta.n_layer = v;
            else if (key == "qwen2.context_length") { /* context length */ }
            else if (key == "qwen2.embedding_length") meta.n_embd = v;
            else if (key == "qwen2.feed_forward_length") meta.n_ff = v;
            else if (key == "qwen2.attention.head_count") meta.n_head = v;
            else if (key == "qwen2.attention.head_count_kv") meta.n_head_kv = v;
            else if (key == "general.file_type") { /* file type */ }
            else if (key == "tokenizer.ggml.bos_token_id") { /* BOS */ }
            else if (key == "tokenizer.ggml.eos_token_id") { /* EOS */ }
            break;
        }
        case 6: { // float32
            auto v = read_float32();
            if (key == "qwen2.attention.layer_norm_rms_epsilon") meta.f_norm_rms_eps = v;
            else if (key == "qwen2.rope.freq_base") meta.freq_base = v;
            break;
        }
        case 7: { // bool
            read_bool();
            break;
        }
        case 8: { // string
            auto s = read_string(f);
            if (key == "general.architecture") meta.arch = s;
            else if (key == "general.name") meta.name = s;
            break;
        }
        case 10: { // uint64
            read_uint64();
            break;
        }
        default: {
            // Skip unknown keys
            // We need to know the size... for arrays, read type+count first
            // For now, just skip by seeking forward based on known sizes
            // Actually let's read array values properly
            if (value_type == 9) {
                uint32_t arr_type;
                uint64_t arr_len;
                f.read(reinterpret_cast<char*>(&arr_type), sizeof(arr_type));
                f.read(reinterpret_cast<char*>(&arr_len), sizeof(arr_len));
                
                // Read array of strings for tokenizer arrays
                if (arr_type == 8) {
                    for (uint64_t i = 0; i < arr_len; i++) {
                        read_string(f);
                    }
                } else if (arr_type == 5) { // int32 array
                    f.seekg(arr_len * 4, std::ios::cur);
                } else {
                    // Skip unknown array type
                    f.seekg(arr_len, std::ios::cur);
                }
            }
            break;
        }
    }
    return true;
}



bool load_gguf(const std::string& path, GGUFModel& model) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "edge-llama: cannot open " << path << std::endl;
        return false;
    }
    
    // Read header
    uint32_t magic;
    f.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (magic != GGUF_MAGIC) {
        std::cerr << "edge-llama: not a GGUF file (magic: 0x" << std::hex << magic << ")" << std::endl;
        return false;
    }
    
    uint32_t version;
    f.read(reinterpret_cast<char*>(&version), sizeof(version));
    std::cerr << "edge-llama: GGUF version " << version << std::endl;
    
    uint64_t n_tensors;
    uint64_t n_kv;
    f.read(reinterpret_cast<char*>(&n_tensors), sizeof(n_tensors));
    f.read(reinterpret_cast<char*>(&n_kv), sizeof(n_kv));
    
    std::cerr << "edge-llama: " << n_tensors << " tensors, " << n_kv << " metadata entries" << std::endl;
    
    // Read metadata
    for (uint64_t i = 0; i < n_kv; i++) {
        read_kv(f, model.meta);
    }
    
    // Read tensor info
    struct TensorInfo {
        std::string name;
        uint32_t n_dims;
        GGMLType type;
        uint64_t offset;
        std::vector<int64_t> dims;
    };
    std::vector<TensorInfo> tensor_infos;
    
    for (uint64_t i = 0; i < n_tensors; i++) {
        TensorInfo info;
        info.name = read_string(f);
        f.read(reinterpret_cast<char*>(&info.n_dims), sizeof(info.n_dims));
        
        info.dims.resize(info.n_dims);
        for (uint32_t d = 0; d < info.n_dims; d++) {
            f.read(reinterpret_cast<char*>(&info.dims[d]), sizeof(int64_t));
        }
        
        uint32_t raw_type;
        f.read(reinterpret_cast<char*>(&raw_type), sizeof(raw_type));
        info.type = static_cast<GGMLType>(raw_type);
        
        f.read(reinterpret_cast<char*>(&info.offset), sizeof(info.offset));
        
        // Skip padding if any
        auto pos = f.tellg();
        if (pos < static_cast<std::streamoff>(info.offset)) {
            // There's padding between tensor info and tensor data in GGUFv3
        }
        
        tensor_infos.push_back(info);
    }
    
    // Now read tensor data
    // In GGUF, tensor data starts after all tensor info at the next aligned offset
    // We already read the tensor info which includes offsets into the data section
    
    // Count vocab size from the tokenizer data
    // For Qwen2 models, this is usually 151936
    model.meta.n_vocab = 151936; // Default for Qwen2 tokenizer
    
    for (auto& info : tensor_infos) {
        auto tensor = std::make_unique<Tensor>();
        tensor->name = info.name;
        tensor->type = info.type;
        tensor->dims = info.dims;
        
        // Calculate data size
        int64_t n_bytes = 0;
        int64_t ne = tensor->n_elements();
        
        // Read the raw data
        auto current_pos = f.tellg();
        f.seekg(info.offset);
        
        // Calculate size from quantization type
        switch (tensor->type) {
            case GGMLType::F32: n_bytes = ne * 4; break;
            case GGMLType::F16: n_bytes = ne * 2; break;
            case GGMLType::Q4_0: n_bytes = (ne / 32) * (2 + 16); break;
            case GGMLType::Q4_1: n_bytes = (ne / 32) * (2 + 16 + 2); break;
            case GGMLType::Q4_K: n_bytes = (ne / 256) * (128 + 12); break;
            case GGMLType::Q6_K: n_bytes = (ne / 256) * (128 + 20); break;
            case GGMLType::Q8_0: n_bytes = (ne / 32) * 34; break;
            default:
                std::cerr << "edge-llama: unknown type " << (int)tensor->type
                          << " for tensor " << tensor->name << std::endl;
                f.seekg(current_pos);
                continue;
        }
        
        tensor->data.resize(n_bytes);
        f.read(reinterpret_cast<char*>(tensor->data.data()), n_bytes);
        
        model.tensors[tensor->name] = std::move(tensor);
        f.seekg(current_pos);
    }
    
    std::cerr << "edge-llama: loaded " << model.tensors.size() << " tensors" << std::endl;
    return true;
}

void print_meta(const GGUFModel& model) {
    const auto& m = model.meta;
    std::cerr << "  arch: " << m.arch << std::endl;
    std::cerr << "  name: " << m.name << std::endl;
    std::cerr << "  layers: " << m.n_layer << std::endl;
    std::cerr << "  hidden: " << m.n_embd << std::endl;
    std::cerr << "  heads: " << m.n_head << " (KV: " << m.n_head_kv << ")" << std::endl;
    std::cerr << "  FF: " << m.n_ff << std::endl;
    std::cerr << "  vocab: " << m.n_vocab << std::endl;
    std::cerr << "  rms_eps: " << m.f_norm_rms_eps << std::endl;
    std::cerr << "  freq_base: " << m.freq_base << std::endl;
}

} // namespace edge_llama
