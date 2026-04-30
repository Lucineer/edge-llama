# edge-llama v0.4.0 — ggml-based CPU inference
#
# Links against ggml-base.so + ggml-cpu.so + ggml.so (backend loader only).
# Uses ggml compute graph for all operations (SIMD, multi-threaded).
# No CUDA dependency. No ollama.
#
# When CMA=1024M after reboot, add ggml-cuda.so for GPU offload.

cmake_minimum_required(VERSION 3.10)
project(edge_llama VERSION 0.4.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# Paths to pip-installed ggml libs (CPU only — NO libggml-cuda)
set(GGML_PREFIX "/home/lucineer/.local/lib/python3.10/site-packages/llama_cpp")
set(GGML_INCLUDE_DIR "/home/lucineer/.local/lib/python3.10/site-packages/include")
set(GGML_LIB_DIR "${GGML_PREFIX}/lib")

include_directories(${GGML_INCLUDE_DIR})

# Source files
set(SOURCES
    src/gguf_loader.cpp
    src/ggml_inference.cpp
    src/tokenizer.cpp
    src/server.cpp
    src/main.cpp
)

add_executable(edge_llama ${SOURCES})

target_link_directories(edge_llama PRIVATE ${GGML_LIB_DIR})
target_link_libraries(edge_llama
    ggml           # backend loader (just init/detect, NO CUDA)
    ggml-cpu       # CPU backend (SIMD, threading)
    ggml-base      # base ops
    pthread
    dl
    stdc++
    m
)

install(TARGETS edge_llama DESTINATION bin)
