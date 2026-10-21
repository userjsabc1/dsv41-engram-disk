#!/usr/bin/env python3
"""Fast memory watchdog: SIGKILL the vLLM process tree when MemAvailable is low.

Polls every 100 ms (the previous bash guard polled every 1 s and used SIGTERM,
which was too slow/weak when memory fell 80 GiB in 5 s)."""
import os
import signal
import sys
import time

THRESHOLD_GIB = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
LOG = os.path.expanduser("~/models/logs/memguard2.log")
ME = os.getpid()
MARKS = ("vllm serve", "VLLM::", "EngineCore", "multiprocessing.spawn")


def mem_available_gib() -> float:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) / 1048576
    return 1e9


def victims() -> list[int]:
    out = []
    for pid in filter(str.isdigit, os.listdir("/proc")):
        pid = int(pid)
        if pid == ME:
            continue
        try:
            if os.stat(f"/proc/{pid}").st_uid != os.getuid():
                continue
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmd = f.read().replace(b"\0", b" ").decode(errors="ignore")
        except OSError:
            continue
        if any(m in cmd for m in MARKS) and "memguard" not in cmd:
            out.append(pid)
    return out


def log(msg: str) -> None:
    with open(LOG, "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")


log(f"guard started, threshold {THRESHOLD_GIB} GiB")
low, last_log, tick = 1e9, 0.0, 0
while True:
    a = mem_available_gib()
    low = min(low, a)
    now = time.time()
    if now - last_log >= 5:
        log(f"MemAvailable={a:.1f} GiB  min={low:.1f} GiB")
        last_log = now
    if a < THRESHOLD_GIB:
        pids = victims()
        log(f"GUARD TRIGGERED a={a:.2f} GiB -> SIGKILL {len(pids)} procs {pids[:12]}")
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        log(f"after kill: MemAvailable={mem_available_gib():.1f} GiB")
        break
    time.sleep(0.1)
