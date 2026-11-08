"""When does the first robot collision happen? (share by step 5 / 25 / 50, median)
  python3 analysis/collision_timing.py logs/control_oa logs/watcher_oa_gentle logs/watcher_oa_medium logs/watcher_oa_strict
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from metrics import load  # noqa: E402

for d in sys.argv[1:]:
    R, name = load(d), os.path.basename(d.rstrip("/"))
    for tier in (None, 0, 1, 2):
        c = sorted(r["first_violation_robot_step"] for r in R
                   if r["first_violation_robot_step"] >= 0 and (tier is None or r["task_index"] // 5 == tier))
        n = len(c)
        if not n:
            continue
        f = lambda k: 100 * sum(1 for x in c if x <= k) / n
        lab = "all" if tier is None else f"L{tier}"
        print(f"{name:18s} {lab:3s} collisions {n:3d}   by step 5: {f(5):3.0f}%   by 25: {f(25):3.0f}%   by 50: {f(50):3.0f}%   median step {c[n // 2]}")
    print()
