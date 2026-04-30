// test_edge.cpp — Test the edge-cuda shared library
// Build: g++ -o test_edge test_edge.cpp -L/home/lucineer/edge-llama/build -ledge-cuda
// Run:   LD_LIBRARY_PATH=/home/lucineer/edge-llama/build ./test_edge model.gguf

#include "edge-cuda.h"
#include <cstdio>

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s model.gguf [prompt]\n", argv[0]);
        return 1;
    }
    
    fprintf(stderr, "edge test: loading %s...\n", argv[1]);
    edge_t* e = edge_load(argv[1]);
    if (!e) {
        fprintf(stderr, "FAILED TO LOAD MODEL\n");
        return 1;
    }
    
    fprintf(stderr, "  backend:   %s\n", edge_backend(e));
    fprintf(stderr, "  layers:    %d\n", edge_n_layer(e));
    fprintf(stderr, "  heads:     %d\n", edge_n_head(e));
    fprintf(stderr, "  embd:      %d\n", edge_n_embd(e));
    fprintf(stderr, "  vocab:     %d\n", edge_n_vocab(e));
    fprintf(stderr, "  vram:      %ld MB / %ld MB\n",
            edge_vram_free(e) / 1048576,
            edge_vram_total(e) / 1048576);
    
    const char* prompt = argc >= 3 ? argv[2] : "Hello from the edge.";
    fprintf(stderr, "\n  generate(\"%s\", 50)...\n", prompt);
    
    int32_t out_len, new_tokens;
    char* result = edge_generate(e, prompt, 50, &out_len, &new_tokens);
    
    fprintf(stderr, "  output:   \"%s\"\n", result ? result : "(null)");
    fprintf(stderr, "  tokens:   %d\n", new_tokens);
    fprintf(stderr, "  speed:    %d t/s\n", edge_tokens_per_second(e));
    
    edge_free_string(result);
    edge_unload(e);
    
    fprintf(stderr, "\nedge: DONE — CUDA inference ready for Plato integration\n");
    return 0;
}
