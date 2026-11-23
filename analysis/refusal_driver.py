"""What made the watcher refuse, and do pauses clear by themselves?
Pure standard library (Mac).

  python3 analysis/refusal_driver.py logs/watcher_oa_medium logs/watcher_oa_stac_medium logs/watcher_oa_strict \
                                     logs/watcher_rs_medium logs/watcher_rs_strict

For each refused episode, look at the REFUSE look (decision 'R') and split its
score into the STAC part (w*stac) and the stuck part ((1-w)*stuck). Also: after a
PAUSE, how often is the very next look a PROCEED (the alarm cleared on re-sampling)?
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from metrics import load  # noqa: E402

for d in sys.argv[1:]:
    rows, name = load(d), os.path.basename(d.rstrip("/"))
    ref = stuck_led = 0
    stac_part, stuck_part, stuck_raw = [], [], []
    pauses = cleared = 0
    for r in rows:
        w = (r.get("monitor") or {}).get("weight", 0.6)
        dec, st, sk = r.get("monitor_decisions", ""), r.get("monitor_stac", []), r.get("monitor_stuck", [])
        for i, c in enumerate(dec[:-1]):
            if c == "p":
                pauses += 1
                cleared += dec[i + 1] == "."
        if "R" in dec:
            i = dec.index("R")
            if i < len(st) and i < len(sk):
                ref += 1
                a, b = w * st[i], (1 - w) * sk[i]
                stac_part.append(a); stuck_part.append(b); stuck_raw.append(sk[i])
                stuck_led += b > a
    m = lambda v: sum(v) / len(v) if v else float("nan")
    print(f"{name:24s} refusals {ref:3d}: stuck-led {stuck_led:3d} ({100*stuck_led/max(ref,1):3.0f}%)  "
          f"mean parts at REFUSE: STAC {m(stac_part):.3f} / stuck {m(stuck_part):.3f} (raw stuck {m(stuck_raw):.2f})   "
          f"| pauses {pauses:5d}, cleared on next look {100*cleared/max(pauses,1):3.0f}%")
