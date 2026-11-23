"""Thresholds for the EXPLORATORY STAC-only watcher (weight 1.0), from the same
held-out calibration run and the same percentiles as the frozen design.

  python3 scripts/calibrate_stac.py logs/calib_oa     # writes configs/thresholds_stac.json

Uses each call's logged STAC value (monitor_stac), skipping the first call as
calibrate.py does. Never touches the frozen configs/thresholds.json.
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from calibrate import PCTS, pct  # noqa: E402

run, out = sys.argv[1], "configs/thresholds_stac.json"
if os.path.exists(out):
    raise SystemExit(f"{out} exists and is frozen. Not overwriting.")
rows = [json.loads(l) for p in glob.glob(os.path.join(run, "part_w*.jsonl")) for l in open(p) if l.strip()]
assert rows and all(r.get("arm") == "calibration" for r in rows), "need calibration-arm rows only"
vals = sorted(v for r in rows for v in r["monitor_stac"][1:])
res = {"frozen": True, "exploratory": "STAC-only watcher (weight 1.0); decided 2026-11-22 AFTER seeing the OA results",
       "source_run": os.path.basename(run.rstrip("/")), "n_episodes": len(rows), "n_values": len(vals),
       "percentiles": PCTS, "thresholds": {k: round(pct(vals, q), 6) for k, q in PCTS.items()},
       "stac_min_median_max": [round(vals[0], 6), round(pct(vals, 50), 6), round(vals[-1], 6)]}
json.dump(res, open(out, "w"), indent=2)
print(json.dumps(res, indent=2))
