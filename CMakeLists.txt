project(edge_cuda)
cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Don't link ggml-cuda — the CUDA init crashes with depleted CMA
# We'll support CUDA via a separate library that dynamically loads it
set(GGML_LIB "/home/lucineer/.local/lib/python3.10/site-packages/llama_cpp/lib")
set(CMAKE_INSTALL_RPATH "${GGML_LIB}")
set(CMAKE_BUILD_RPATH "${GGML_LIB}")

set(GGML_INCLUDE "/home/lucineer/.local/lib/python3.10/site-packages/include")
include_directories(${GGML_INCLUDE})
link_directories(${GGML_LIB})

# Shared library — CPU only for now (CUDA after reboot)
add_library(edge_cuda SHARED
    src/edge-cuda.c
    src/edge-cuda-impl.cpp
)
target_link_libraries(edge_cuda
    llama ggml ggml-cpu ggml-base
    pthread dl stdc++ m
)
set_target_properties(edge_cuda PROPERTIES
    OUTPUT_NAME "edge-cuda"
    VERSION "1.0.0"
    SOVERSION "1"
)

# Test binary
add_executable(edge_test
    src/edge-cuda.c
    src/edge-cuda-impl.cpp
)
target_link_libraries(edge_test
    llama ggml ggml-cpu ggml-base
    pthread dl stdc++ m
)
