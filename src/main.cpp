#include "gguf_loader.h"
#include "model_qwen2.h"
#include "server.h"
#include <iostream>
#include <cstring>
#include <thread>
#include <chrono>

// edge-llama v0.3.0 — Pure CPU inference for Jetson
// Links directly against ggml-cpu + ggml-base (no CUDA)
//
// Usage:
//   ./edge_llama <model.gguf>                # Interactive mode
//   ./edge_llama <model.gguf> serve <socket> # Unix socket server
//   ./edge_llama <model.gguf> tcp <port>     # TCP server

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.gguf> [serve <socket>|tcp <port>]" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    std::string mode = (argc > 2) ? argv[2] : "load";
    
    std::cerr << "edge-llama v0.3.0" << std::endl;
    std::cerr << "  model: " << model_path << std::endl;
    std::cerr << "  backend: CPU-only (no CUDA)" << std::endl;

    // Load GGUF metadata
    edge_llama::GGUFModel gguf_model;
    if (!edge_llama::load_gguf(model_path, gguf_model)) {
        std::cerr << "edge-llama: failed to load GGUF file" << std::endl;
        return 1;
    }
    
    edge_llama::print_meta(gguf_model);
    std::cerr << "  tensors: " << gguf_model.tensors.size() << std::endl;

    // Initialize model (dequantize weights to f32)
    edge_llama::Qwen2Model model;
    if (!model.init(gguf_model)) {
        std::cerr << "edge-llama: model initialization failed" << std::endl;
        return 1;
    }

    if (mode == "load") {
        // Interactive mode
        std::string input;
        std::cerr << "edge-llama: interactive. Type prompts, /quit to exit." << std::endl;
        while (true) {
            std::cerr << "> ";
            if (!std::getline(std::cin, input)) break;
            if (input == "/quit" || input == "/exit") break;
            if (input.empty()) continue;

            auto output = model.generate(input, 64);
            std::cout << output << std::endl;
            
            auto s = model.get_stats();
            std::cerr << "[gen: " << s.total_tokens << " tok | "
                      << s.tokens_per_second << " t/s]" << std::endl;
        }
    }
    else if (mode == "serve" && argc > 3) {
        std::string socket_path = argv[3];
        edge_llama::Server server(model);
        if (!server.start(socket_path)) {
            std::cerr << "edge-llama: failed to start server" << std::endl;
            return 1;
        }
        while (true) {
            server.serve();
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    else if (mode == "tcp" && argc > 3) {
        int port = std::stoi(argv[3]);
        edge_llama::Server server(model);
        if (!server.start_tcp(port)) {
            std::cerr << "edge-llama: failed to start TCP server" << std::endl;
            return 1;
        }
        while (true) {
            server.serve();
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    return 0;
}
