#include "tokenizer.h"
#include "gguf_loader.h"
#include "backend_inference.h"
#include <iostream>
#include <cstring>

using namespace edge_llama;

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " <model.gguf> [command]\n";
    std::cerr << "\nCommands:\n";
    std::cerr << "  (none)       Interactive mode — type prompts, /quit to exit\n";
    std::cerr << "  serve <sock> Unix socket server at <path>\n";
    std::cerr << "  tcp <port>   TCP server on port\n";
    std::cerr << "  test         Load model, validate, print stats\n";
    std::cerr << "\nEnvironment:\n";
    std::cerr << "  GGML_CUDA=0  Force CPU only\n";
}

// Interactive mode
static void interactive(BackendInference& inf) {
    std::cout << "edge-llama: interactive. Type prompts, /quit to exit." << std::endl;
    std::cout << "> " << std::flush;
    
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line == "/quit" || line == "/q") break;
        if (line.empty()) {
            std::cout << "> " << std::flush;
            continue;
        }
        
        auto result = inf.generate(line);
        
        auto stats = inf.get_stats();
        std::cout << result << std::endl;
        std::cout << "\n[backend: " << stats.backend_name 
                  << " | prompt: " << stats.prompt_tokens 
                  << " | gen: " << stats.total_tokens 
                  << " | " << stats.tokens_per_second << " t/s]" << std::endl;
        std::cout << "> " << std::flush;
    }
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {
    std::cerr << "edge-llama v0.5.0 — ggml backend inference" << std::endl;
    
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }
    
    const char* model_path = argv[1];
    
    // Check for CUDA override
    bool force_cpu = false;
    if (const char* env = std::getenv("GGML_CUDA")) {
        if (env[0] == '0') force_cpu = true;
    }
    
    // Load GGUF
    std::cerr << "edge-llama: loading " << model_path << "..." << std::endl;
    GGUFModel model;
    if (!load_gguf(model_path, model)) {
        std::cerr << "edge-llama: failed to load model" << std::endl;
        return 1;
    }
    
    print_meta(model);
    
    // Initialize inference engine
    BackendInference inf;
    if (!inf.init(model)) {
        std::cerr << "edge-llama: model initialization failed" << std::endl;
        return 1;
    }
    
    // Dispatch
    if (argc >= 3) {
        if (strcmp(argv[2], "test") == 0) {
            std::cerr << "edge-llama: model loaded and validated" << std::endl;
            return 0;
        } else if (strcmp(argv[2], "serve") == 0 && argc >= 4) {
            std::cerr << "edge-llama: server mode not yet implemented" << std::endl;
            return 0;
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }
    
    interactive(inf);
    
    return 0;
}
