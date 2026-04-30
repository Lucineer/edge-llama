# edge-llama

**Self-contained C++ inference server for Jetson AGX Orin.**

No ollama dependency. Pure C++, no Python. Links directly against ggml for CPU or CUDA compute.

## Status: MVP Complete — Blocked on CMA

### What Works ✅
- **GGUF v3 file loading** — full metadata parsing, all 339 tensors loaded
- **Q4_K, Q6_K, F32 dequantization** — on-the-fly and bulk
- **Full Qwen2 transformer architecture** — 28-layer attention + FFN
- **C++ server** — Unix socket and TCP serving
- **No ollama dependency** — standalone binary (79KB)
- **No CUDA required** for load/dequant — works on depleted CMA

### What's Blocked ❌
- **GPU inference** — CMA pool depleted to 644KB / 512MB by NVIDIA driver
  - `cma=1024M` already set in `/boot/extlinux/extlinux.conf`, needs reboot
  - After reboot: ollama will get 1024MB CMA → CUDA works → edge-llama GPU mode
- **CPU inference** — naive matmul in model_qwen2.cpp (6.5B ops/token @ ~0.5 GOPS)
  - Would take 2-3 seconds per token on ARM64 CPU
  - Could use ggml compute graph but ggml.so has CUDA baked in
  - **Fix**: compile our own CPU-only ggml from source (needs time on Jetson)

### The Path Forward

1. **Reboot** → `cma=1024M` → CUDA works → edge-llama in GPU mode
2. **Or compile CPU-only ggml** from source with `GGML_CUDA=OFF`
3. **Either way**: edge-llama becomes a 79KB CUDA/C++ inference server
4. **Then**: flato MUD links edge-llama as shared library

## Build

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
```

## Usage

```bash
# Interactive mode
./edge_llama path/to/model.gguf

# Unix socket server (for flato MUD)
./edge_llama path/to/model.gguf serve /tmp/edge-llama.sock

# TCP server
./edge_llama path/to/model.gguf tcp 8080
```

## Architecture

```
src/
  gguf_loader.h/cpp     — GGUF v3 file parser (metadata + tensor info)
  ggml_ops.h/cpp        — Naive CPU ops (matmul, norm, rope, silu)
  model_qwen2.h/cpp     — Qwen2 transformer inference loop
  server.h/cpp          — Unix socket + TCP server
  main.cpp              — Entry point
```

## Design Philosophy

**The wedge.** edge-llama is the first brick in the wall. It's:
- Minimal (79KB binary)
- Self-contained (no Python, no ollama, no CUDA at link time)
- Composable (flato MUD links it as a library)
- Cross-platform (C++17, runs on any POSIX system with ggml)

From here: flato (Fleet Plato MUD in C) → Mesh protocol → Edge-cloud continuum.
