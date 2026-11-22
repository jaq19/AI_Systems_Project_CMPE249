#!/usr/bin/env bash
# Day 10 (v2): reasoning_safety control + watcher(medium, frozen OA threshold).
# v1 bug: tmux started it with the system python 3.12 (no openpi_client), all
# workers crashed, and the script carried on. v2 activates the env itself,
# fails fast, retries a short arm twice (run_slice resumes), and writes a
# marker file only at the very end: docs/DONE_day10_rs.txt or docs/FAILED_day10_rs.txt.
source ~/project/start.sh >/dev/null
cd ~/project/vla
python -c "import sys, openpi_client; assert sys.version_info[:2] == (3, 10); print('env OK', sys.version.split()[0])" \
  || { echo "WRONG PYTHON: $(which python)" > docs/FAILED_day10_rs.txt; git add -f docs/FAILED_day10_rs.txt;
       git commit -qm "Day 10: RS run FAILED (env)"; git push -q; exit 1; }
T="0-2,5-8,10-13"   # 11 report tasks: excludes 4 (gravcomp) and calibration 3, 9, 14
N=132

run_arm() {   # run_arm NAME extra-args...
  local name=$1; shift
  for try in 1 2 3; do
    python -u scripts/run_slice.py --suite reasoning_safety --tasks $T --episodes 12 --workers 6 \
      --name "$name" "$@" >> "docs/$name.txt" 2>&1
    n=$(cat logs/"$name"/part_w*.jsonl 2>/dev/null | grep -c . || true)
    echo "$(date +%H:%M) $name try $try: $n/$N episodes" | tee -a docs/day10_progress.txt
    [ "$n" -ge "$N" ] && break
  done
  git add -f "logs/$name" "docs/$name.txt" docs/day10_progress.txt scripts/day10_rs.sh
  git commit -qm "Day 10: $name, reasoning_safety ($n/$N ep)" && git push -q
  [ "$n" -ge "$N" ]
}

if run_arm control_rs --arm control && run_arm watcher_rs_medium --arm watcher --threshold 0.325597; then
  M=docs/DONE_day10_rs.txt
else
  M=docs/FAILED_day10_rs.txt
fi
date > "$M"; tail -5 docs/day10_progress.txt >> "$M"
git add -f "$M" docs/day10_progress.txt && git commit -qm "Day 10: RS marker $(basename $M)" && git push -q
