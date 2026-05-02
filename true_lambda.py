"""
true_lambda.py — Serverless Inference Dispatch (fleet-innovations #6)

Routes each inference request to the fastest available backend:
1. Native inference (libedge-cuda.so) — fastest if model is loaded
2. Ollama inference — good for models not in native
3. Gateways via Unix socket — alternate edge nodes

Designed as a drop-in replacement for edge-gateway's model routing.
"""

import json
import time
import socket
import subprocess
import threading
from typing import Optional, Callable

# Backend definitions
BACKENDS = {}

_lambda_lock = threading.Lock()


def register_backend(name: str, priority: int, 
                     check_fn: Callable, 
                     invoke_fn: Callable):
    """Register a dispatch backend.
    
    Args:
        name: Backend name (e.g., 'native', 'ollama', 'socket')
        priority: Lower number = attempted first (1=fastest)
        check_fn: Returns True if this backend is available
        invoke_fn: Callable(prompt, max_tokens) -> dict
    """
    BACKENDS[name] = {
        "priority": priority,
        "check": check_fn,
        "invoke": invoke_fn,
        "name": name,
    }


def dispatch(prompt: str, max_tokens: int = 128,
             preferred: str = None) -> dict:
    """Dispatch inference to best available backend.
    
    Args:
        prompt: Input text
        max_tokens: Maximum tokens to generate
        preferred: If set, only try this backend
    
    Returns:
        dict with text, tokens, tps, backend, latency_ms
    """
    sorted_backends = sorted(
        BACKENDS.items(),
        key=lambda x: x[1]["priority"]
    )
    
    if preferred:
        sorted_backends = [b for b in sorted_backends if b[0] == preferred]
    
    errors = []
    
    for name, backend in sorted_backends:
        start = time.time()
        try:
            if not backend["check"]():
                continue
            
            result = backend["invoke"](prompt, max_tokens)
            elapsed = (time.time() - start) * 1000
            
            result["backend"] = name
            result["latency_ms"] = round(elapsed, 1)
            return result
        except Exception as e:
            errors.append(f"{name}: {e}")
            continue
    
    return {
        "text": f"[all backends failed: {'; '.join(errors)}]",
        "tokens": 0,
        "tps": 0,
        "backend": "none",
        "latency_ms": 0,
    }


def benchmark_all(prompt: str = "hello", max_tokens: int = 10) -> list:
    """Benchmark every available backend. Returns sorted results."""
    results = []
    for name, backend in sorted(BACKENDS.items(), key=lambda x: x[1]["priority"]):
        try:
            if not backend["check"]():
                results.append({"backend": name, "available": False})
                continue
            
            start = time.time()
            result = backend["invoke"](prompt, max_tokens)
            elapsed = time.time() - start
            
            results.append({
                "backend": name,
                "available": True,
                "latency_s": round(elapsed, 2),
                "tps": result.get("tps", 0),
                "tokens": result.get("tokens", 0),
            })
        except Exception as e:
            results.append({"backend": name, "available": False, "error": str(e)})
    
    return results


# =============================================================
#  Backend Implementations
# =============================================================

def _check_native():
    """Check if native inference backend is available."""
    try:
        # Quick check via health endpoint
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(1)
        s.connect("/tmp/edge-native.sock")
        s.send(json.dumps({"prompt": "ping", "max_tokens": 1}).encode() + b"\n")
        resp = s.recv(4096)
        s.close()
        return b"text" in resp
    except Exception:
        return False


def _invoke_native(prompt: str, max_tokens: int) -> dict:
    """Send request via Unix socket to native inference."""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(30)
    s.connect("/tmp/edge-native.sock")
    req = json.dumps({"prompt": prompt, "max_tokens": max_tokens}) + "\n"
    s.send(req.encode())
    resp = b""
    while True:
        chunk = s.recv(4096)
        if not chunk:
            break
        resp += chunk
        if b"\n" in resp:
            break
    s.close()
    return json.loads(resp.decode().strip())


def _check_ollama():
    """Check if Ollama is available."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1)
        s.connect(("127.0.0.1", 11434))
        s.close()
        return True
    except Exception:
        return False


def _invoke_ollama(prompt: str, max_tokens: int) -> dict:
    """Send request via HTTP to Ollama."""
    import urllib.request
    
    body = json.dumps({
        "model": "deepseek-r1:1.5b",
        "prompt": prompt,
        "options": {"num_predict": max_tokens},
        "stream": False,
    }).encode()
    
    req = urllib.request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read())
    
    return {
        "text": result.get("response", ""),
        "tokens": result.get("eval_count", 0),
        "tps": result.get("eval_count", 0) / max(result.get("eval_duration", 1) / 1e9, 0.01),
    }


def _check_gateway():
    """Check if edge gateway is available."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1)
        s.connect(("127.0.0.1", 11435))
        s.close()
        return True
    except Exception:
        return False


def _invoke_gateway(prompt: str, max_tokens: int) -> dict:
    """Send request via HTTP to edge gateway (forced native)."""
    import urllib.request
    
    body = json.dumps({
        "model": "deepseek-r1:1.5b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
    }).encode()
    
    req = urllib.request.Request(
        "http://127.0.0.1:11435/v1/chat/completions?native=true",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read())
    
    return {
        "text": result.get("choices", [{}])[0].get("message", {}).get("content", ""),
        "tokens": result.get("usage", {}).get("completion_tokens", 0),
        "tps": result.get("tps", 0),
    }


# Register backends at module import time
register_backend("native", 1, _check_native, _invoke_native)
register_backend("ollama", 2, _check_ollama, _invoke_ollama)
register_backend("gateway", 3, _check_gateway, _invoke_gateway)
