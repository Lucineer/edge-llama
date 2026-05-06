#!/usr/bin/env python3
"""
warp_room.py — Warp-as-Room Tile Classifier (CPU Simulation)
V2: Bigram features + TF weighting

Port of SuperInstance/gpu-native-room-inference to CPU via numpy.
GPU warp -> numpy vector ops. Room collective -> batch embedding.

Works now on Jetson CPU. Swap to cupy when CUDA comes back.

Usage:
  python3 warp_room.py                      # Classify recent tiles
  python3 warp_room.py --train              # Learn room profiles from tiles
  python3 warp_room.py --daemon             # Continuous classification loop
  python3 warp_room.py --infer "text here"  # Classify a single query
"""

import json, os, sys, time, hashlib, argparse
from pathlib import Path

try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False

PLATO_URL = os.environ.get("PLATO_URL", "http://localhost:8847")

DEFAULT_ROOM_PROFILES = {
    "edge": {
        "keywords": ["jetson", "cpu", "gpu", "memory", "temperature", "load", "uptime",
                      "disk", "thermal", "fan", "power", "nvidia", "cuda", "nvcc",
                      "arm64", "aarch64", "swap", "network", "interface", "sensor",
                      "telemetry", "hardware", "clock", "throttle"],
        "priority": 0,
    },
    "research": {
        "keywords": ["research", "paper", "study", "findings", "analysis", "experiment",
                      "benchmark", "performance", "test", "comparison", "evaluation",
                      "learn", "training", "dataset", "model", "inference", "llm",
                      "neural", "embedding", "vector", "similarity", "tile",
                      "investigation", "methodology", "result", "conclusion"],
        "priority": 1,
    },
    "fleet": {
        "keywords": ["fleet", "agent", "oracle", "forge", "vessel", "bottle", "matrix",
                      "heartbeat", "sync", "mesh", "iron", "coordination", "bridge",
                      "pki", "cert", "trust", "deadman", "migration", "protocol",
                      "lighthouse", "beacon", "dm", "conduit"],
        "priority": 2,
    },
    "jc1": {
        "keywords": ["jc1", "jetsonclaw", "plato", "evennia", "flato", "mythos", "cocapn",
                      "edge-llama", "libllama", "gguf", "sovereign vessel",
                      "captain log", "bottle from", "jc1 research", "jc1 fleet",
                      "jc1 edge", "jc1 telemetry", "jc1 system"],
        "priority": 3,
    },
}


class WarpRoomClassifier:
    """CPU simulation of warp-as-room inference. V2 with bigrams + TF."""

    def __init__(self, profiles=None, use_bigrams=True):
        self.profiles = profiles or DEFAULT_ROOM_PROFILES.copy()
        self.use_bigrams = use_bigrams
        self.vocab, self.bigram_vocab = self._build_vocabulary()
        self.room_vectors = self._build_room_vectors()
        self.learned_vectors = {}
        self.learn_count = 0

    def _build_vocabulary(self):
        words = set()
        bigrams = set()
        for name, profile in self.profiles.items():
            for kw in profile["keywords"]:
                parts = kw.lower().split()
                for p in parts:
                    words.add(p)
                if len(parts) >= 2 and self.use_bigrams:
                    for i in range(len(parts) - 1):
                        bigrams.add(f"{parts[i]} {parts[i+1]}")
        return sorted(words), sorted(bigrams)

    def _text_features(self, text):
        """Extract TF-weighted features from text."""
        text_lower = text.lower()
        words = text_lower.split()
        total_dim = len(self.vocab) + len(self.bigram_vocab)
        vec = np.zeros(total_dim, dtype=np.float32)

        for i, word in enumerate(self.vocab):
            c = text_lower.count(word)
            if c > 0:
                vec[i] = np.sqrt(c)

        offset = len(self.vocab)
        if self.use_bigrams and len(words) >= 2:
            text_bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
            for i, bg in enumerate(self.bigram_vocab):
                c = text_bigrams.count(bg)
                if c > 0:
                    vec[offset + i] = np.sqrt(c)

        return vec

    def _build_room_vectors(self):
        total_dim = len(self.vocab) + len(self.bigram_vocab)
        vectors = {}
        for name, profile in self.profiles.items():
            text = " ".join(profile["keywords"])
            vec = self._text_features(text)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vectors[name] = vec
        return vectors

    def classify(self, text, threshold=0.15):
        vec = self._text_features(text)
        norm = np.linalg.norm(vec)
        if norm == 0:
            return {"room": "unknown", "confidence": 0.0, "scores": {},
                    "label": "No features found"}

        vec = vec / norm
        scores = {}
        for name, rvec in self.room_vectors.items():
            sim = float(np.dot(vec, rvec))
            pb = self.profiles.get(name, {}).get("priority", 0) * 0.05
            sim = min(1.0, sim + pb)
            scores[name] = round(sim, 4)

        best = max(scores, key=scores.get)
        best_score = scores[best]

        if best_score < threshold:
            return {"room": "unknown", "confidence": best_score, "scores": scores,
                    "label": f"Best ({best}: {best_score}) below {threshold}"}

        return {"room": best, "confidence": best_score, "scores": scores,
                "label": f"Classified as '{best}' ({best_score})"}

    def train(self, text, room_label):
        vec = self._text_features(text)
        norm = np.linalg.norm(vec)
        if norm == 0:
            return False
        vec = vec / norm
        total_dim = len(self.vocab) + len(self.bigram_vocab)
        if room_label not in self.learned_vectors:
            self.learned_vectors[room_label] = np.zeros(total_dim, dtype=np.float32)
        lr = 0.3
        self.learned_vectors[room_label] = (1 - lr) * self.learned_vectors[room_label] + lr * vec
        if room_label in self.room_vectors:
            self.room_vectors[room_label] = 0.7 * self.room_vectors[room_label] + 0.3 * self.learned_vectors[room_label]
            n = np.linalg.norm(self.room_vectors[room_label])
            if n > 0:
                self.room_vectors[room_label] /= n
        self.learn_count += 1
        return True

    def get_room_stats(self):
        stats = {}
        for name, profile in self.profiles.items():
            total_dim = len(self.vocab) + len(self.bigram_vocab)
            stats[name] = {
                "keywords": len(profile["keywords"]),
                "vector_norm": round(float(np.linalg.norm(
                    self.room_vectors.get(name, np.zeros(1)))), 4),
                "vocab_dim": total_dim,
                "learned": name in self.learned_vectors,
                "priority": profile.get("priority", 0),
            }
        return stats

    def to_dict(self):
        return {
            "type": "warp_room_classifier_v2",
            "version": 2,
            "learn_count": self.learn_count,
            "vocab_size": len(self.vocab),
            "bigram_size": len(self.bigram_vocab),
            "rooms": list(self.profiles.keys()),
            "profiles": self.profiles,
        }


# --- PLATO Integration ---

def fetch_tiles(limit=50):
    try:
        import urllib.request
        req = urllib.request.Request(f"{PLATO_URL}/tiles/recent?limit={limit}")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {"tiles": []}


# --- CLI ---

def main():
    parser = argparse.ArgumentParser(description="Warp-as-Room Tile Classifier")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--daemon", action="store_true")
    parser.add_argument("--infer", type=str)
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args()

    if not HAVE_NUMPY:
        print("ERROR: numpy required. pip install numpy")
        sys.exit(1)

    classifier = WarpRoomClassifier()

    if args.infer:
        result = classifier.classify(args.infer)
        print(json.dumps(result, indent=2))
        return

    if args.stats:
        stats = classifier.get_room_stats()
        print(json.dumps(stats, indent=2))
        return

    if args.train:
        data = fetch_tiles(200)
        tiles = data.get("tiles", data if isinstance(data, list) else [])
        count = 0
        for tile in tiles:
            text = f"{tile.get('question','')} {tile.get('answer','')} {tile.get('tags',[])}"
            room = tile.get("room", tile.get("domain", "unknown"))
            if room in classifier.profiles:
                if classifier.train(text, room):
                    count += 1
        print(f"trained on {count}/{len(tiles)} tiles")
        stats = classifier.get_room_stats()
        for name, s in stats.items():
            print(f"  {name}: dim={s['vocab_dim']}, norm={s['vector_norm']}, "
                  f"kw={s['keywords']}, learned={s['learned']}")
        return

    if args.daemon:
        print(f"warp-room daemon — {PLATO_URL}")
        while True:
            try:
                data = fetch_tiles(100)
                tiles = data.get("tiles", data if isinstance(data, list) else [])
                unclassed = [t for t in tiles if not t.get("classification")]
                for t in unclassed[:20]:
                    text = f"{t.get('question','')} {t.get('answer','')} {t.get('tags',[])}"
                    classifier.classify(text)
                if unclassed:
                    print(f"{time.strftime('%H:%M:%S')} checked {len(unclassed)} tiles")
                time.sleep(60)
            except KeyboardInterrupt:
                print("\nshutdown")
                break
            except Exception as e:
                print(f"err: {e}")
                time.sleep(60)
        return

    # Default: classify
    data = fetch_tiles(50)
    tiles = data.get("tiles", data if isinstance(data, list) else [])
    print(f"Classifying {len(tiles)} tiles...")
    for tile in tiles:
        text = f"{tile.get('question','')} {tile.get('answer','')} {tile.get('tags',[])}"
        result = classifier.classify(text)
        r, c = result["room"], result["confidence"]
        q = tile.get('question','')[:50]
        print(f"  [{r:<10} {c:.2f}] {q}")


if __name__ == "__main__":
    main()
