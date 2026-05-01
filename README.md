# edge-llama 🚀

**Native LLM inference on Jetson — no Ollama, no cloud, just shared libraries.**

Links `libllama.so` (llama.cpp) directly into your process. Loads GGUF models, generates text at 19 t/s on CPU, and can be embedded into MUDs, gateways, or mesh agents via a 51KB shared library (`libedge-cuda.so`).

## Quick Start

```bash
# Build
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4

# Run test (points to your deepseek-r1 GGUF)
./edge_test

# Start the interactive MUD on port 4003
./flato 4003 /tmp/edge-native.sock
```

## Status: v0.6.0 — Production CPU Inference ✅

### Works
- **Native CPU inference** at 19 t/s on deepseek-r1:1.5b (Q4_K_M, 1.04GB GGUF)
- **`libedge-cuda.so`** — 51KB shared library, links `libllama.so` directly
- **`edge_native.py`** — Python ctypes wrapper (`with EdgeModel("model.gguf") as e: e.generate(...)`)
- **Streaming API** — callback-based per-token generation (`edge_generate_stream`)
- **Python singleton** — `EdgePlatoModel` loaded once, re-entrant via threading.Lock
- **flato MUD** — C17 telnet server on :4003 with `/think`, `/status`, `/gpu`, `/cuda`, `/peers`
- **Edge gateway integration** — `?native=true` routes through native inference at 18 t/s
- **SSE streaming** through OpenAI-compatible `/v1/chat/completions?native=true&stream=true`

### Architecture

```
Your App (Python/Evennia/C)
        │
        ▼
libedge-cuda.so  (51KB shared library)
        │
        ▼
libllama.so      (llama.cpp C API — GGUF, tokenize, sample, generate)
        │
        ▼
    model.gguf    (deepseek-r1:1.5b, ~1.0GB on disk)
```

No HTTP. No subprocess. No Python loops. Just `dlopen()` and call.

### Commands

| Command | Where | What |
|---------|-------|------|
| `@infer <prompt>` | Evennia (Plato MUD) | Native inference, streaming |
| `@think <prompt>` | Evennia | Ship AI — ship persona response |
| `/think <prompt>` | flato (telnet :4003) | Same, via Unix socket |
| `/gpu` | flato | nvidia-smi status |
| `/cuda` | flato | CUDA toolkit + device + CMA info |
| `?native=true` | Edge gateway (:11435) | Native backend instead of Ollama |

### Blocked (GPU)
- **GPU inference** — CMA pool depleted (NVIDIA driver allocates during first CUDA call, never frees). Workaround: `CUDA_VISIBLE_DEVICES=""` for CPU-only. Fix: reboot with `cma=1024M`.

## Why Not Just Use Ollama?

Ollama is great. edge-llama is different:

| | Ollama | edge-llama |
|---|---|---|
| **API** | HTTP (subprocess) | Shared library (in-process) |
| **Latency** | ~50ms overhead | ~2μs function call |
| **Embedding** | Web server | `dlopen()` into your app |
| **Fleet agent** | Separate daemon | Link into MUD process |
| **Binary size** | ~200MB | 51KB |

The edge gateway (`edge-gateway.py`) uses both: Ollama by default, native fallback when Ollama is down, forced native via `?native=true`.

## Build Dependencies

- C++17 compiler (GCC or Clang)
- `libllama.so` — installable via `pip install llama-cpp-python` (includes .so)
- CUDA 12.6 (optional — auto-detected at runtime)

## Files

```
build/libedge-cuda.so      # Shared library — 51KB
build/edge_test            # Test binary
edge_native.py             # Python ctypes wrapper
flato.c                    # C17 MUD telnet server
src/
  edge-cuda.h              # Public API header
  edge-cuda.c              # C stub → C++ impl
  edge-cuda-impl.cpp       # Links libllama.so, calls llama_eval
```

## Design

**The wedge.** edge-llama is the first brick in the wall. From here: full GPU inference (after reboot), mesh-native embeddings, in-MUD training.
