#!/usr/bin/env python3
"""
edge_native.py — Python ctypes wrapper for libedge-cuda.so

Loads the shared library directly into the Python process.
No subprocess, no HTTP server — just pure shared library calls.
Implements context manager for automatic cleanup.

Usage:
    from edge_native import Edge
    with Edge("/path/to/model.gguf") as model:
        text = model.generate("Hello world", max_tokens=100)
        print(f"Generated {model.tps} t/s: {text}")
"""

import ctypes
import ctypes.util
import os
import sys
from pathlib import Path
from typing import Optional

# Find the shared library
LIB_PATHS = [
    os.path.expanduser("~/edge-llama/build/libedge-cuda.so"),
    os.path.expanduser("~/edge-llama/build/libedge-cuda.so.1"),
]

class EdgeModel:
    """
    Native edge inference via shared library.
    Loads libedge-cuda.so and keeps model in process memory.
    
    Thread-safe: all generate() calls are serialized by a mutex in the C code.
    """
    
    def __init__(self, model_path: str):
        self.lib = None
        self.impl = None
        self.model_path = model_path
        
    def __enter__(self):
        self.load()
        return self
        
    def __exit__(self, *args):
        self.unload()
        
    def load(self):
        """Load the shared library and model."""
        # Find and load the library
        lib_path = None
        for p in LIB_PATHS:
            if os.path.exists(p):
                lib_path = str(Path(p).resolve())
                break
        
        if not lib_path:
            # Search LD_LIBRARY_PATH
            found = ctypes.util.find_library("edge-cuda")
            if found:
                lib_path = found
            else:
                raise FileNotFoundError(
                    "libedge-cuda.so not found. Build it first:\n"
                    "  cd ~/edge-llama && mkdir -p build && cd build && cmake .. && make"
                )
        
        self.lib = ctypes.CDLL(lib_path)
        
        # Set up function signatures
        # edge_load: (const char*) -> opaque_ptr
        self.lib.edge_load.restype = ctypes.c_void_p
        self.lib.edge_load.argtypes = [ctypes.c_char_p]
        
        # edge_unload: (opaque_ptr) -> void
        self.lib.edge_unload.restype = None
        self.lib.edge_unload.argtypes = [ctypes.c_void_p]
        
        # edge_generate: (opaque_ptr, const char*, int32_t, int32_t*, int32_t*) -> char*
        self.lib.edge_generate.restype = ctypes.POINTER(ctypes.c_char)
        self.lib.edge_generate.argtypes = [
            ctypes.c_void_p,      # impl
            ctypes.c_char_p,      # prompt
            ctypes.c_int32,       # max_tokens
            ctypes.POINTER(ctypes.c_int32),  # out_len
            ctypes.POINTER(ctypes.c_int32),  # new_tokens
        ]
        
        # edge_free_string: (char*) -> void
        self.lib.edge_free_string.restype = None
        self.lib.edge_free_string.argtypes = [ctypes.c_void_p]
        
        # Accessors
        self.lib.edge_n_layer.restype = ctypes.c_int32
        self.lib.edge_n_layer.argtypes = [ctypes.c_void_p]
        self.lib.edge_n_embd.restype = ctypes.c_int32
        self.lib.edge_n_embd.argtypes = [ctypes.c_void_p]
        self.lib.edge_n_head.restype = ctypes.c_int32
        self.lib.edge_n_head.argtypes = [ctypes.c_void_p]
        self.lib.edge_n_vocab.restype = ctypes.c_int32
        self.lib.edge_n_vocab.argtypes = [ctypes.c_void_p]
        self.lib.edge_backend.restype = ctypes.c_char_p
        self.lib.edge_backend.argtypes = [ctypes.c_void_p]
        self.lib.edge_tokens_per_second.restype = ctypes.c_int32
        self.lib.edge_tokens_per_second.argtypes = [ctypes.c_void_p]
        self.lib.edge_vram_total.restype = ctypes.c_int64
        self.lib.edge_vram_total.argtypes = [ctypes.c_void_p]
        self.lib.edge_vram_free.restype = ctypes.c_int64
        self.lib.edge_vram_free.argtypes = [ctypes.c_void_p]
        
        # Load model
        model_path_bytes = self.model_path.encode('utf-8')
        self.impl = self.lib.edge_load(model_path_bytes)
        
        if not self.impl:
            error_fn = getattr(self.lib, 'edge_last_error', None)
            if error_fn:
                error_fn.restype = ctypes.c_char_p
                error_fn.argtypes = []
                err = error_fn()
                raise RuntimeError(f"Failed to load model: {err.decode() if err else 'unknown'}")
            else:
                raise RuntimeError(f"Failed to load model: {self.model_path}")
    
    def unload(self):
        """Unload the model and free memory."""
        if self.lib and self.impl:
            self.lib.edge_unload(self.impl)
            self.impl = None
    
    def generate(self, prompt: str, max_tokens: int = 128) -> str:
        """
        Generate text from prompt.
        Returns the generated text (not including the prompt).
        """
        if not self.impl:
            raise RuntimeError("Model not loaded")
        
        prompt_bytes = prompt.encode('utf-8')
        out_len = ctypes.c_int32(0)
        new_tokens = ctypes.c_int32(0)
        
        result_ptr = self.lib.edge_generate(
            self.impl,
            prompt_bytes,
            ctypes.c_int32(max_tokens),
            ctypes.byref(out_len),
            ctypes.byref(new_tokens)
        )
        
        if not result_ptr:
            return ""
        
        try:
            text = ctypes.string_at(result_ptr, out_len.value).decode('utf-8', errors='replace')
            return text
        finally:
            self.lib.edge_free_string(result_ptr)
    
    @property
    def n_layer(self) -> int:
        return self.lib.edge_n_layer(self.impl) if self.impl else 0
        
    @property
    def n_embd(self) -> int:
        return self.lib.edge_n_embd(self.impl) if self.impl else 0
        
    @property
    def n_head(self) -> int:
        return self.lib.edge_n_head(self.impl) if self.impl else 0
        
    @property
    def n_vocab(self) -> int:
        return self.lib.edge_n_vocab(self.impl) if self.impl else 0
        
    @property
    def backend(self) -> str:
        if not self.impl:
            return "none"
        return self.lib.edge_backend(self.impl).decode()
        
    @property
    def tps(self) -> int:
        return self.lib.edge_tokens_per_second(self.impl) if self.impl else 0
        
    @property
    def vram_total(self) -> int:
        return self.lib.edge_vram_total(self.impl) if self.impl else 0
        
    @property
    def vram_free(self) -> int:
        return self.lib.edge_vram_free(self.impl) if self.impl else 0
    
    def __str__(self) -> str:
        return (f"Edge[backend={self.backend}, layers={self.n_layer}, "
                f"heads={self.n_head}, embd={self.n_embd}, "
                f"vocab={self.n_vocab}, tps={self.tps}]")


# ── CLI Test ──
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Edge native inference")
    parser.add_argument("model", help="Path to GGUF model file")
    parser.add_argument("prompt", nargs="?", default="Hello.", help="Prompt text")
    parser.add_argument("--tokens", "-n", type=int, default=64, help="Max tokens")
    args = parser.parse_args()
    
    print(f"Loading {args.model}...")
    with EdgeModel(args.model) as edge:
        print(f"Model: {edge}")
        print(f"Prompt: {args.prompt}")
        print(f"Generating...")
        result = edge.generate(args.prompt, max_tokens=args.tokens)
        print(f"\nResult ({edge.tps} t/s):")
        print(result)
