#!/usr/bin/env python3
"""Synthesize PLATO tiles using native edge inference and submit back."""
import urllib.request
import json
import sys
import os

sys.path.insert(0, '/home/lucineer/edge-llama')
from edge_native import EdgeModel

def get_tiles():
    """Fetch all unique tiles from all rooms."""
    rooms = ['edge', 'research', 'jc1', 'fleet']
    all_tiles = []
    for room in rooms:
        try:
            resp = urllib.request.urlopen(f"http://localhost:8847/room/{room}", timeout=5)
            data = json.loads(resp.read())
            tiles = data.get('tiles', [data]) if isinstance(data, dict) else data
            if isinstance(tiles, list):
                seen = {}
                for t in tiles:
                    q = t.get('question', '') or t.get('title', '')
                    if q and q not in seen:
                        seen[q] = t
                all_tiles.extend(list(seen.values()))
        except:
            pass
    return all_tiles

tiles = get_tiles()
if not tiles:
    print("ERROR: Could not fetch any tiles")
    sys.exit(1)

# Group by room
by_room = {}
for t in tiles:
    r = t.get('room', 'unknown')
    if r not in by_room:
        by_room[r] = []
    by_room[r].append(t)

print(f"Total unique tiles: {len(tiles)}")
for r, ts in sorted(by_room.items()):
    print(f"  {r}: {len(ts)}")

# Build a dense summary from the tile contents
fact_lines = []
for r in ['edge', 'research']:
    for t in by_room.get(r, []):
        q = t.get('question', t.get('title', ''))
        a = t.get('answer', t.get('content', ''))
        conf = t.get('confidence', 0.5)
        if isinstance(a, str) and len(a) > 10:
            fact_lines.append(f"[{r}][conf={conf:.1f}] Q: {q}\nA: {a[:200]}")

facts = '\n'.join(fact_lines[:20])
print(f"\nFacts for synthesis ({len(fact_lines)} entries):\n{facts[:500]}...\n")

# Generate synthesis
prompt = (
    "<｜begin▁of▁sentence｜>You are JC1, the Jetson hardware specialist on the Lucineer fleet. "
    f"Below are {len(tiles)} PLATO knowledge tiles collected from monitoring and research.\n\n"
    f"TILE DATA:\n{facts}\n\n"
    "Write a concise 3-paragraph status report:\n"
    "1) What patterns do you see in the telemetry?\n"
    "2) What needs attention or action?\n"
    "3) What should be built next?\n"
    "Be direct. Use hardware language. No greetings.<｜end▁of▁sentence｜>\n"
    "\n"
    "JC1 STATUS REPORT:\n"
)

print(f"Prompt: {len(prompt)} chars, generating...")

model = EdgeModel('/home/lucineer/edge-llama/models/dsr1-1.5b-q4km.gguf')
model.load()
result = model.generate(prompt, max_tokens=384)
text = result.decode() if isinstance(result, bytes) else result

print("\n" + "="*60)
print("JC1 STATUS REPORT (Native Inference)")
print("="*60)
print(text)
print("="*60)

# Submit result back as a tile
synthesis_tile = {
    "room": "research",
    "domain": "synthesis",
    "question": f"JC1 Tile Synthesis — {len(tiles)} tiles analyzed",
    "answer": text.strip(),
    "agent": "jc1-native-inference",
    "confidence": 0.7
}
try:
    req = urllib.request.Request(
        "http://localhost:8847/submit",
        data=json.dumps(synthesis_tile).encode(),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    resp = urllib.request.urlopen(req, timeout=5)
    print(f"\n✅ Synthesis submitted as tile to research room")
except Exception as e:
    print(f"\n❌ Submit failed: {e}")

model.unload()
