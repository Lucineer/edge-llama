#!/usr/bin/env python3
"""
sensor-pipeline.py — JC1 Hardware Sensor → PLATO Tile Pipeline

Feeds live Jetson hardware telemetry into plato-server as structured tiles.
Based on: SuperInstance/sensor-plato-bridge + SuperInstance/marine-gpu-edge concepts.
Runs every 5 minutes via systemd timer.
"""

import json, os, subprocess, time, urllib.request, urllib.error
from pathlib import Path

PLATO_URL = os.environ.get("PLATO_URL", "http://localhost:8847")
AGENT_NAME = "sensor-pipeline"
TAGS = ["sensor", "hardware", "jetson", "jc1"]

def run_cmd(cmd):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        return r.stdout.strip(), r.stderr.strip(), r.returncode
    except subprocess.TimeoutExpired:
        return "", "TIMEOUT", -1

def collect_telemetry():
    """Collect hardware telemetry — returns list of tile dicts."""
    tiles = []
    ts = int(time.time())
    
    # 1. CPU / Memory
    with open("/proc/loadavg") as f:
        load = f.read().strip().split()[:3]
    with open("/proc/meminfo") as f:
        mem = {}
        for line in f:
            k, v = line.split(":")
            mem[k.strip()] = int(v.strip().split()[0])
    
    mem_used = mem.get("MemTotal", 0) - mem.get("MemFree", 0) - mem.get("Buffers", 0) - mem.get("Cached", 0)
    mem_pct = round(mem_used / mem.get("MemTotal", 1) * 100, 1)
    
    tiles.append({
        "domain": "edge",
        "question": f"What is the Jetson load average at {ts}?",
        "answer": f"1min: {load[0]}, 5min: {load[1]}, 15min: {load[2]}",
        "tags": ["sensor", "cpu", "load"] + TAGS,
        "room": "edge",
    })
    
    tiles.append({
        "domain": "edge",
        "question": f"What is the Jetson memory status at {ts}?",
        "answer": f"Total: {mem.get('MemTotal',0)}kB, Used: {mem_used}kB ({mem_pct}%), Free: {mem.get('MemFree',0)}kB, Swap: {mem.get('SwapTotal',0)}kB",
        "tags": ["sensor", "memory", "swap"] + TAGS,
        "room": "edge",
    })
    
    # 2. Disk
    disk = subprocess.run("df -h /", shell=True, capture_output=True, text=True, timeout=5)
    disk_line = disk.stdout.strip().split("\n")[-1]
    parts = disk_line.split()
    tiles.append({
        "domain": "edge",
        "question": f"What is the Jetson disk usage at {ts}?",
        "answer": f"Filesystem: {parts[0]}, Size: {parts[1]}, Used: {parts[2]}, Avail: {parts[3]}, Use%: {parts[4]}",
        "tags": ["sensor", "disk", "storage"] + TAGS,
        "room": "edge",
    })
    
    # 3. Uptime
    uptime_secs = time.time() - os.path.getmtime("/proc/1/cmdline")
    uptime_h = uptime_secs / 3600
    with open("/proc/uptime") as f:
        up_secs = float(f.read().split()[0])
    
    tiles.append({
        "domain": "edge",
        "question": f"What is the Jetson uptime at {ts}?",
        "answer": f"{up_secs/3600:.1f} hours ({up_secs:.0f} seconds)",
        "tags": ["sensor", "uptime"] + TAGS,
        "room": "edge",
    })
    
    # 4. Thermal (Jetson-specific)
    temp = subprocess.run("cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null", 
        shell=True, capture_output=True, text=True, timeout=5)
    temps = [t.strip() for t in temp.stdout.split() if t.strip()]
    if temps:
        thermal_info = "; ".join([f"Zone {i}: {int(t)/1000:.1f}C" for i, t in enumerate(temps[:6])])
        tiles.append({
            "domain": "edge",
            "question": f"What are the Jetson temperatures at {ts}?",
            "answer": thermal_info,
            "tags": ["sensor", "thermal", "temperature"] + TAGS,
            "room": "edge",
        })
    
    # 5. Process count
    procs = subprocess.run("ps aux | wc -l", shell=True, capture_output=True, text=True, timeout=5)
    tiles.append({
        "domain": "edge",
        "question": f"How many processes are running on Jetson at {ts}?",
        "answer": f"{procs.stdout.strip()} processes",
        "tags": ["sensor", "processes"] + TAGS,
        "room": "edge",
    })
    
    # 6. Network
    net = subprocess.run("cat /proc/net/dev | grep -E 'eth|wlan|end' | head -3", 
        shell=True, capture_output=True, text=True, timeout=5)
    for line in net.stdout.strip().split("\n"):
        if line:
            iface = line.split(":")[0].strip()
            stats = line.split()[1:]
            rx_bytes = stats[0]
            tx_bytes = stats[8]
            tiles.append({
                "domain": "edge",
                "question": f"What is the {iface} network status at {ts}?",
                "answer": f"Interface {iface} — RX: {rx_bytes} bytes, TX: {tx_bytes} bytes",
                "tags": ["sensor", "network", iface] + TAGS,
                "room": "edge",
            })
    
    return tiles

def submit_tiles(tiles):
    accepted = 0
    for tile in tiles:
        data = json.dumps(tile).encode()
        try:
            req = urllib.request.Request(f"{PLATO_URL}/submit",
                data=data, headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=10) as resp:
                r = json.loads(resp.read())
                if r.get("status") == "accepted":
                    accepted += 1
        except Exception as e:
            pass
    return accepted

def main():
    tiles = collect_telemetry()
    n = submit_tiles(tiles)
    print(f"sensor-pipeline: {n}/{len(tiles)} tiles submitted to {PLATO_URL}")

if __name__ == "__main__":
    main()
