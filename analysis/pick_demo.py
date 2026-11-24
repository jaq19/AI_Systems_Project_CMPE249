"""Pick start states for the demo video (pure standard library, Mac).
  python3 analysis/pick_demo.py
CAUGHT: harmful-instruction start states where control engaged the object but the
        strict watcher avoided it and refused. MISSED: obstacle start states where
        both arms hit an obstacle early and the strict watcher had not refused yet.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from metrics import load  # noqa: E402

key = lambda r: (r["task_index"], r["episode_index"])
c_rs = {key(r): r for r in load("logs/control_rs")}
caught = [(k, c_rs[k], w) for w in load("logs/watcher_rs_strict") for k in [key(w)]
          if k in c_rs and c_rs[k]["violation_robot"] and not w["violation_robot"] and w["refused"]]
print("CAUGHT (harmful instructions; control harmed, strict avoided and refused):")
for k, c, w in sorted(caught, key=lambda x: x[2]["refused_at"])[:6]:
    print(f"  task {k[0]:2d} ep {k[1]:2d}  control harm @ {c['first_violation_robot_step']:3d}  strict refused @ {w['refused_at']:3d}  \"{c['instruction']}\"")
c_oa = {key(r): r for r in load("logs/control_oa")}
missed = [(k, c_oa[k], w) for w in load("logs/watcher_oa_strict") for k in [key(w)]
          if k in c_oa and c_oa[k]["violation_robot"] and 0 <= w["first_violation_robot_step"] <= 50
          and (not w["refused"] or w["refused_at"] > w["first_violation_robot_step"])]
print("\nMISSED (obstacles; both arms collided, strict collided by step 50 before any refusal):")
for k, c, w in sorted(missed, key=lambda x: x[2]["first_violation_robot_step"])[:6]:
    print(f"  task {k[0]:2d} ep {k[1]:2d}  control hit @ {c['first_violation_robot_step']:3d}  strict hit @ {w['first_violation_robot_step']:3d}  \"{c['instruction']}\"")
print(f"\n{len(caught)} caught candidates, {len(missed)} missed candidates")
