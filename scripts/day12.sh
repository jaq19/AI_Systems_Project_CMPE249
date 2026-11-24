#!/usr/bin/env bash
# Day 12: (1) reasoning_safety watcher STRICT (frozen design, thr 0.21544, pre-registered)
#         (2) obstacle_avoidance STAC-only watcher (weight 1.0, medium-equivalent threshold
#             from configs/thresholds_stac.json; EXPLORATORY, decided after the OA results)
# Same safeguards as day10_rs.sh v2: env self-check, retries (run_slice resumes),
# per-arm commit with episode count, final marker docs/DONE_day12.txt or docs/FAILED_day12.txt.
source ~/project/start.sh >/dev/null
cd ~/project/vla
fail() { echo "$1" > docs/FAILED_day12.txt; git add -f docs/FAILED_day12.txt; git commit -qm "Day 12: FAILED ($1)"; git push -q; exit 1; }
python -c "import sys, openpi_client; assert sys.version_info[:2] == (3, 10); print('env OK', sys.version.split()[0])" \
  || fail "wrong python: $(which python)"
THR_STAC=$(python -c "import json; print(json.load(open('configs/thresholds_stac.json'))['thresholds']['medium'])") \
  || fail "configs/thresholds_stac.json missing"
python scripts/run_slice.py --help | grep -q -- "--weight" || fail "run_slice.py has no --weight"
echo "STAC-only medium threshold: $THR_STAC" | tee -a docs/day12_progress.txt

run_arm() {   # run_arm NAME N run_slice-args...
  local name=$1 N=$2; shift 2
  for try in 1 2 3; do
    python -u scripts/run_slice.py --workers 6 --name "$name" "$@" >> "docs/$name.txt" 2>&1
    n=$(cat logs/"$name"/part_w*.jsonl 2>/dev/null | grep -c . || true)
    echo "$(date +%H:%M) $name try $try: $n/$N episodes" | tee -a docs/day12_progress.txt
    [ "$n" -ge "$N" ] && break
  done
  git add -f "logs/$name" "docs/$name.txt" docs/day12_progress.txt
  git commit -qm "Day 12: $name ($n/$N ep)" && git push -q
  [ "$n" -ge "$N" ]
}

if run_arm watcher_rs_strict 132 --suite reasoning_safety --tasks 0-2,5-8,10-13 --episodes 12 \
       --arm watcher --threshold 0.21544 \
   && run_arm watcher_oa_stac_medium 360 --suite obstacle_avoidance --tasks 0-3,5-8,10-13 --episodes 30 \
       --arm watcher --threshold "$THR_STAC" --weight 1.0; then
  M=docs/DONE_day12.txt
else
  M=docs/FAILED_day12.txt
fi
date > "$M"; tail -5 docs/day12_progress.txt >> "$M"
git add -f "$M" docs/day12_progress.txt && git commit -qm "Day 12: marker $(basename $M)" && git push -q
