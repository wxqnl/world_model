#!/usr/bin/env python3
from __future__ import annotations

import os
import signal
import subprocess


PATTERNS = (
    "build_window_geom_tar_shards_v1.py",
    "tar --format=gnu --null -C /0604-10T-test/wm3d_v5/cache/vggt_window_geom",
)


def _ppid(pid: int) -> int | None:
    try:
        text = open(f"/proc/{pid}/stat", "r", encoding="utf-8", errors="ignore").read()
        tail = text.rsplit(")", 1)[1].strip().split()
        return int(tail[1])
    except Exception:
        return None


def _ancestor_pids(pid: int) -> set[int]:
    out: set[int] = set()
    cur = _ppid(pid)
    while cur and cur > 1 and cur not in out:
        out.add(cur)
        cur = _ppid(cur)
    return out


def main() -> None:
    self_pid = os.getpid()
    protected = {self_pid} | _ancestor_pids(self_pid)
    out = subprocess.check_output(["ps", "-eo", "pid=,cmd="], text=True, errors="ignore")
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        pid_s, _, cmd = line.partition(" ")
        try:
            pid = int(pid_s)
        except ValueError:
            continue
        if pid in protected:
            continue
        if any(pattern in cmd for pattern in PATTERNS):
            print(f"kill {pid} {cmd[:180]}", flush=True)
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


if __name__ == "__main__":
    main()
