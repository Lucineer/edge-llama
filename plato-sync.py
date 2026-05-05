#!/usr/bin/env python3
"""
plato-sync.py — JC1 Evennia ↔ plato-server tile sync daemon

Bridges JC1's Evennia MUD knowledge tiles to the local plato-server,
and pulls fleet tiles from plato-server into Evennia's world/ directory.

Architecture:
    Evennia (world/*.md) ──→ plato-sync.py ──→ plato-server (:8847)
                                   ↑
                           fleet tiles come back
                           via plato-server's Matrix sync

Run: systemd timer every 5 minutes
"""

import os
import sys
import json
import time
import glob
import urllib.request
import urllib.error
from pathlib import Path

# ── Config ──────────────────────────────────────────────────
PLATO_URL = os.environ.get("PLATO_URL", "http://localhost:8847")
EVENNIA_WORLD = os.environ.get("EVENNIA_WORLD_DIR",
    "/home/lucineer/plato-jetson/world")
AGENT_NAME = os.environ.get("SYNC_AGENT", "jc1")
DOMAIN_TAG = os.environ.get("DOMAIN_TAG", "jc1")
SYNC_INTERVAL = int(os.environ.get("SYNC_INTERVAL", "300"))  # 5 min

# ── Tile extraction from Evennia world/ ─────────────────────

def extract_tiles_from_world():
    """
    Parse Evennia world/*.md files for YAML front-matter tiles.
    Format:
        ---
        domain: research
        room: bridge
        ---
        # Title

        Question? → Answer.
    """
    tiles = []
    seen_hashes = set()
    
    world_dir = Path(EVENNIA_WORLD)
    if not world_dir.exists():
        print(f"  world dir not found: {EVENNIA_WORLD}")
        return tiles
    
    for md_file in sorted(world_dir.glob("*.md")):
        name = md_file.stem
        if name.startswith("_") or name == "README":
            continue
            
        content = md_file.read_text()
        
        # Extract YAML front matter
        domain = 'jc1'
        room = 'bridge'
        
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                front = parts[1]
                body = parts[2].strip()
                
                # Parse domain and room from front matter
                for line in front.split("\n"):
                    if line.startswith("domain:"):
                        domain = line.split(":", 1)[1].strip()
                    elif line.startswith("room:"):
                        room = line.split(":", 1)[1].strip()
                
                # Create tile from file
                # Use first heading as question
                title = ""
                for line in body.split("\n"):
                    line = line.strip()
                    if line.startswith("# "):
                        title = line[2:].strip()
                        break
                
                if title:
                    tile_hash = hash(title + domain + room)
                    if tile_hash not in seen_hashes:
                        seen_hashes.add(tile_hash)
                        tiles.append({
                            "domain": domain,
                            "question": title,
                            "answer": body[:500],  # truncate
                            "agent": AGENT_NAME,
                            "source": f"evennia:{name}"
                        })
    
    return tiles


def get_fleet_tiles():
    """Fetch recent tiles from plato-server for import."""
    try:
        url = f"{PLATO_URL}/tiles/recent"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and "tiles" in data:
                return data["tiles"]
            return []
    except Exception as e:
        print(f"  fetch error: {e}")
        return []


def submit_tile(tile):
    """Submit a single tile to plato-server."""
    try:
        data = json.dumps(tile).encode()
        req = urllib.request.Request(
            f"{PLATO_URL}/submit",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            result = json.loads(resp.read())
            return result.get("status") == "accepted"
    except Exception as e:
        print(f"  submit error for {tile.get('question','?')[:30]}: {e}")
        return False


def import_tile_to_evennia(tile, tile_id):
    """Write a fleet tile as a new Evennia world file."""
    world_dir = Path(EVENNIA_WORLD)
    if not world_dir.exists():
        return False
    
    question = tile.get("question", "Untitled")
    domain = tile.get("domain", "fleet")
    answer = tile.get("answer", "")
    
    # Sanitize filename
    safe_name = "".join(c if c.isalnum() or c in " -_" else "_" for c in question[:40])
    safe_name = safe_name.strip().replace(" ", "-").lower()
    fname = world_dir / f"fleet-{safe_name}.md"
    
    if fname.exists():
        return False  # skip duplicates
    
    content = f"""---
domain: {domain}
room: fleet
source: plato-server
tile_id: {tile_id}
---

# {question}

{answer}

*Imported from fleet PLATO at {time.strftime("%Y-%m-%d %H:%M:%S")}*
"""
    fname.write_text(content)
    print(f"  imported: {fname.name}")
    return True


def sync():
    """Run one sync cycle."""
    print(f"=== plato-sync tick {int(time.time())} ===")
    
    # Phase 1: Push Evennia tiles to plato-server
    print("Phase 1: Push edge tiles → plato-server")
    tiles = extract_tiles_from_world()
    print(f"  {len(tiles)} tiles extracted from world/")
    
    submitted = 0
    for tile in tiles:
        if submit_tile(tile):
            submitted += 1
    print(f"  {submitted}/{len(tiles)} submitted")
    
    # Phase 2: Pull fleet tiles → Evennia
    print("Phase 2: Pull fleet tiles → edge")
    fleet_tiles = get_fleet_tiles()
    print(f"  {len(fleet_tiles)} tiles available from plato-server")
    
    imported = 0
    for tile in fleet_tiles:
        tile_id = tile.get("tile_id", "")
        if tile_id and "source" not in tile.get("meta", {}):
            if import_tile_to_evennia(tile, tile_id):
                imported += 1
    print(f"  {imported} new tiles imported")
    
    print(f"=== Sync complete ({submitted}↑ {imported}↓) ===")
    return submitted + imported > 0


if __name__ == "__main__":
    sync()
