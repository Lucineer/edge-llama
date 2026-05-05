#!/usr/bin/env python3
"""
JC1 Edge Research Agent — Fork of CCC's research pipeline with local inference.

Role: Fleet edge research assistant. Monitors fleet repos, synthesizes findings,
submits tiles to plato-server, notifies Casey when something significant drops.

Key difference from CCC: CCC runs on K2.5 (cloud API). JC1 runs on edge hardware
with native LLM inference at 18 t/s. No API key needed. Always online.

Research cadence:
- Check repo activity every 5 minutes
- Trigger deeper analysis on significant events
- Submit findings as PLATO tiles
- Drop a brief summary when something matters
"""

import os
import sys
import json
import time
import hashlib
import threading
import subprocess
import re
from pathlib import Path
from datetime import datetime, timezone

# ── Config ──────────────────────────────────────────────────
PLATO_URL = os.environ.get("PLATO_URL", "http://localhost:8847")
AGENT_NAME = os.environ.get("EDGE_AGENT", "jc1-research")
CHECK_INTERVAL = int(os.environ.get("CHECK_INTERVAL", "300"))  # 5 min
GITHUB_TOKEN = os.environ.get("GH_TOKEN", "")  # set via environment

# Repos to monitor (SuperInstance org — fleet's active repos)
MONITORED_REPOS = [
    "SuperInstance/flux-compiler",
    "SuperInstance/flux-runtime-c",
    "SuperInstance/flux-core",
    "SuperInstance/flux-hardware",
    "SuperInstance/flux-cuda",
    "SuperInstance/plato-server",
    "SuperInstance/plato-sdk-unified",
    "SuperInstance/constraint-theory-core",
    "SuperInstance/sensor-plato-bridge",
    "SuperInstance/flux-verify-api",
    "SuperInstance/cocapn-core",
    "SuperInstance/marine-gpu-edge",
    "SuperInstance/holodeck-rust",
    "SuperInstance/flux-os",
    "SuperInstance/fleet-agent",
]

# Edge system data collectors
HARDWARE_METRICS = [
    ("Memory", lambda: open("/proc/meminfo").read().split("\n")[0:3]),
    ("Load", lambda: [str(x) for x in os.getloadavg()]),
    ("Uptime", lambda: open("/proc/uptime").read().strip().split()[0]),
    ("Temperature", lambda: open("/sys/class/thermal/thermal_zone0/temp").read().strip() if os.path.exists("/sys/class/thermal/thermal_zone0/temp") else "N/A"),
]


# ── PLATO Server Client ─────────────────────────────────────

def submit_tile(question, answer, domain="edge", agent=AGENT_NAME):
    """Submit a knowledge tile to the local plato-server."""
    import urllib.request
    
    try:
        data = json.dumps({
            "question": question[:200],
            "answer": answer[:2000],
            "domain": domain,
            "agent": agent,
        }).encode()
        req = urllib.request.Request(
            f"{PLATO_URL}/submit",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            result = json.loads(resp.read())
            return result
    except Exception as e:
        return {"status": "error", "error": str(e)}


def get_recent_tiles(room=None, limit=10):
    """Fetch recent tiles from plato-server."""
    import urllib.request
    
    try:
        if room:
            url = f"{PLATO_URL}/room/{room}"
        else:
            url = f"{PLATO_URL}/tiles/recent"
        
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return []


# ── GitHub Repo Monitoring ──────────────────────────────────

def fetch_gh_api(url):
    """Fetch from GitHub API with auth."""
    import urllib.request
    try:
        req = urllib.request.Request(url)
        req.add_header("Authorization", f"Bearer {GITHUB_TOKEN}")
        req.add_header("Accept", "application/vnd.github.v3+json")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {"error": str(e)}


def check_fleet_activity():
    """Check which fleet repos have recent commits."""
    recent = []
    since = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 86400))
    
    for repo in MONITORED_REPOS:
        url = f"https://api.github.com/repos/{repo}/commits?since={since}&per_page=1"
        data = fetch_gh_api(url)
        if isinstance(data, list) and len(data) > 0:
            sha = data[0].get("sha", "")[:8]
            msg = data[0].get("commit", {}).get("message", "")[:60]
            author = data[0].get("commit", {}).get("author", {}).get("name", "?")
            recent.append(f"{repo}: {sha} {author} — {msg}")
        elif isinstance(data, dict) and "error" in data:
            pass  # skip rate-limited
    return recent


# ── Edge System Telemetry ───────────────────────────────────

def collect_hardware_tiles():
    """Collect system metrics and submit as knowledge tiles."""
    tiles = []
    
    # Memory
    with open("/proc/meminfo") as f:
        meminfo = f.read()
    
    total_match = re.search(r"MemTotal:\s+(\d+) kB", meminfo)
    avail_match = re.search(r"MemAvailable:\s+(\d+) kB", meminfo)
    
    if total_match and avail_match:
        total_mb = int(total_match.group(1)) / 1024
        avail_mb = int(avail_match.group(1)) / 1024
        pct = ((total_mb - avail_mb) / total_mb) * 100
        tiles.append(("JC1 Memory Status", 
            f"Total: {total_mb:.0f}MB, Available: {avail_mb:.0f}MB, Used: {pct:.0f}%"))
    
    # CPU Load
    load = os.getloadavg()
    tiles.append(("JC1 CPU Load", 
        f"1min: {load[0]:.2f}, 5min: {load[1]:.2f}, 15min: {load[2]:.2f}"))
    
    # Uptime
    with open("/proc/uptime") as f:
        uptime_secs = float(f.read().strip().split()[0])
        uptime_str = f"{int(uptime_secs // 86400)}d {int((uptime_secs % 86400) // 3600)}h"
    tiles.append(("JC1 Uptime", f"{uptime_str} ({uptime_secs:.0f}s)"))
    
    # GPU (if available)
    nv_dev = Path("/dev/nvidia0")
    gpu_info = "No GPU detected" if not nv_dev.exists() else "GPU device present at /dev/nvidia0"
    tiles.append(("JC1 GPU Status", gpu_info))
    
    return tiles


# ── Fleet Change Detection ──────────────────────────────────

class FleetMonitor:
    """Tracks repo state across ticks and submits tiles on changes."""
    
    def __init__(self, state_file="/tmp/jc1-fleet-state.json"):
        self.state_file = state_file
        self.state = {"repos": {}, "last_check": 0}
        self._load()
    
    def _load(self):
        try:
            with open(self.state_file) as f:
                self.state = json.load(f)
        except:
            pass
    
    def _save(self):
        with open(self.state_file, "w") as f:
            json.dump(self.state, f)
    
    def check(self):
        """Check for changes since last tick."""
        changes = []
        since = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.state["last_check"]))
        
        for repo in MONITORED_REPOS:
            url = f"https://api.github.com/repos/{repo}/commits?since={since}&per_page=3"
            data = fetch_gh_api(url)
            if not isinstance(data, list) or len(data) == 0:
                continue
            
            prev_sha = self.state["repos"].get(repo, "")
            for commit in data:
                sha = commit.get("sha", "")
                if sha.startswith(prev_sha):
                    break
                msg = commit.get("commit", {}).get("message", "")[:100]
                author = commit.get("commit", {}).get("author", {}).get("name", "?")
                changes.append(f"{repo} — {author}: {msg}")
        
        # Update state
        now = int(time.time())
        self.state["last_check"] = now
        for repo in MONITORED_REPOS:
            url = f"https://api.github.com/repos/{repo}/commits?per_page=1"
            data = fetch_gh_api(url)
            if isinstance(data, list) and len(data) > 0:
                self.state["repos"][repo] = data[0].get("sha", "")[:16]
        self._save()
        
        return changes


# ── Main Research Loop ──────────────────────────────────────

def research_tick():
    """Run one research cycle."""
    print(f"\n=== JC1 Research Tick {int(time.time())} ===")
    
    # 1. Collect hardware telemetry
    print("Phase 1: Hardware telemetry")
    tiles = collect_hardware_tiles()
    for q, a in tiles:
        result = submit_tile(q, a, domain="edge", agent="jc1-telemetry")
        status = result.get("status","?")
        print(f"  {q[:40]}: {result.get('status','?')}")
    
    # 2. Check fleet repo activity
    print("Phase 2: Fleet repo monitoring")
    changes = [f"{r.split('/')[1]} — active" for r in MONITORED_REPOS[:5]]  # simplified
    recent = check_fleet_activity()
    for r in recent[:5]:
        print(f"  {r[:70]}")
    
    if recent:
        summary = "\n".join(recent[:5])
        submit_tile(
            "Fleet repo activity (last 24h)",
            summary,
            domain="fleet",
            agent=AGENT_NAME
        )
    
    # 3. Check for significant changes
    print("Phase 3: Change detection")
    monitor = FleetMonitor()
    changes = monitor.check()
    for c in changes:
        print(f"  ! {c[:70]}")
    
    if changes:
        # Significant change detected — submit analysis
        change_summary = "\n".join(changes)
        submit_tile(
            "Fleet change alert",
            f"Detected {len(changes)} changes since last check:\n{change_summary}",
            domain="fleet",
            agent=AGENT_NAME
        )
    
    # 4. Plato server health
    print("Phase 4: Plato server check")
    try:
        tiles = get_recent_tiles()
        print(f"  plato-server: {len(tiles) if isinstance(tiles, list) else tiles.get('total_tiles',0)} tiles")
    except:
        print("  plato-server unreachable")
    
    print(f"=== Tick complete ===")


def run_loop():
    """Run the research loop continuously."""
    print(f"JC1 Edge Research Agent starting...")
    print(f"PLATO server: {PLATO_URL}")
    print(f"Interval: {CHECK_INTERVAL}s")
    print(f"Monitoring {len(MONITORED_REPOS)} fleet repos")
    
    # First tick immediately
    research_tick()
    
    while True:
        time.sleep(CHECK_INTERVAL)
        research_tick()


if __name__ == "__main__":
    # Single tick if --once flag
    if "--once" in sys.argv:
        research_tick()
    else:
        run_loop()
