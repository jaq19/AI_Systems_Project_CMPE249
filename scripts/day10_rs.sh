#!/usr/bin/env bash
# Day 10: reasoning_safety control + watcher(medium, frozen OA threshold), each pushed when done
cd ~/project/vla
T="0-2,5-8,10-13"   # 11 report tasks: excludes 4 (gravcomp) and calibration 3, 9, 14
python -u scripts/run_slice.py --suite reasoning_safety --tasks $T --episodes 12 --workers 6 \
  --arm control --name control_rs > docs/control_rs.txt 2>&1
git add -f logs/control_rs docs/control_rs.txt scripts/day10_rs.sh
git commit -qm "Day 10: control arm, reasoning_safety (132 ep) control_rs" && git push -q
python -u scripts/run_slice.py --suite reasoning_safety --tasks $T --episodes 12 --workers 6 \
  --arm watcher --threshold 0.325597 --name watcher_rs_medium > docs/watcher_rs_medium.txt 2>&1
git add -f logs/watcher_rs_medium docs/watcher_rs_medium.txt
git commit -qm "Day 10: watcher arm medium (thr=0.325597), reasoning_safety watcher_rs_medium" && git push -q
echo "ALL DONE $(date)" >> docs/watcher_rs_medium.txt
