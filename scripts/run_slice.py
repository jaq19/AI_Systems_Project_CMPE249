"""Run many episodes in parallel worker processes against ONE pi0.5 server.

Why parallel: on Day 3 an episode took ~162 s, of which only ~15 s was model
inference (104 calls x 143 ms). The rest is CPU (MuJoCo physics + rendering),
so N worker processes give close to N x throughput until the vCPUs run out.
The server handles one request at a time, which is fine at this load.

Output: one JSON line per episode in <out>/<run_name>/part_w<k>.jsonl.
Resumable: re-running the same command skips (task, episode) pairs already
written. Partial lines from a killed run are ignored.

Examples:
  python -u scripts/run_slice.py --suite obstacle_avoidance --tasks 0-3 \
      --episodes 2 --workers 3 --name timing_w3
  python -u scripts/run_slice.py --suite obstacle_avoidance --tasks 0-3,5-8,10-13 \
      --episodes 30 --workers 3 --name control_oa_s0
"""
import argparse
import glob
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def parse_tasks(spec):
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def load_rows(run_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(run_dir, "part_w*.jsonl"))):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass                      # half-written line from a kill
    return rows


def _worker(k, jobs, cfg, run_dir):
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"
    sys.path.insert(0, ROOT)
    if os.environ.get("VLA_FAKE_ENV"):        # test hook only (no LIBERO)
        import tests._fake_env as fe
        fe.install()
    from harness.runner import run_episode
    if cfg["dummy"]:
        from harness.policy import DummyPolicy
        policy = DummyPolicy(seed=k)
    else:
        from harness.policy import Pi05Policy
        policy = Pi05Policy()
    path = os.path.join(run_dir, f"part_w{k}.jsonl")
    with open(path, "a") as f:
        for task, ep in jobs:
            r = run_episode(policy, cfg["suite"], task, ep, cfg["seed"],
                            monitor=None, arm=cfg["arm"],
                            threshold=cfg["threshold"],
                            max_env_steps=cfg["max_steps"],
                            terminate_on_violation=cfg["terminate"])
            r["dummy"] = cfg["dummy"]
            r["worker"] = k
            r["run_name"] = cfg["name"]
            f.write(json.dumps(r) + "\n")
            f.flush()
            os.fsync(f.fileno())
            print(f"  [w{k}] task {task:2d} ep {ep:2d}: raw={r['success_raw']!s:5} "
                  f"robotviol@{r['first_violation_robot_step']:4d} "
                  f"knock={r['knocked']!s:5} "
                  f"steps={r['steps']:3d} {r['seconds']:6.1f}s", flush=True)


def summarize(rows, wall=None):
    n = len(rows)
    if n == 0:
        print("  no rows")
        return
    mean = lambda key: sum(bool(r[key]) for r in rows) / n
    secs = sum(r["seconds"] for r in rows) / n
    print(f"  episodes {n}")
    print(f"  success_raw {mean('success_raw'):.0%}   success_paper {mean('success_paper'):.0%}"
          f"   success_paper_robot {mean('success_paper_robot'):.0%}")
    print(f"  violation(robot) [PRIMARY] {mean('violation_robot'):.0%}   "
          f"violation(all pairs) {mean('violation'):.0%}   "
          f"knocked {sum(bool(r.get('knocked')) for r in rows) / n:.0%}")
    print(f"  mean seconds/episode (per worker) {secs:.0f}")
    if wall:
        print(f"  wall {wall:.0f}s  ->  throughput {3600 * n / wall:.1f} episodes/hour")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", required=True)
    p.add_argument("--tasks", required=True, help="e.g. 0-3,5-8,10-13")
    p.add_argument("--episodes", type=int, default=2, help="per task")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--arm", default="control")
    p.add_argument("--threshold", type=float, default=float("nan"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=520)
    p.add_argument("--terminate", action="store_true",
                   help="end episode at first violation (paper rule); default log-only")
    p.add_argument("--dummy", action="store_true")
    p.add_argument("--name", required=True)
    p.add_argument("--out", default=os.path.join(ROOT, "logs"))
    a = p.parse_args()

    run_dir = os.path.join(a.out, a.name)
    os.makedirs(run_dir, exist_ok=True)
    cfg = dict(suite=a.suite, arm=a.arm, threshold=a.threshold, seed=a.seed,
               max_steps=a.max_steps, terminate=a.terminate, dummy=a.dummy,
               name=a.name)
    json.dump({**cfg, "tasks": a.tasks, "episodes": a.episodes,
               "workers": a.workers}, open(os.path.join(run_dir, "config.json"), "w"),
              indent=2)

    done = {(r["task_index"], r["episode_index"]) for r in load_rows(run_dir)}
    jobs = [(t, e) for t in parse_tasks(a.tasks) for e in range(a.episodes)
            if (t, e) not in done]
    print(f"{a.name}: {len(done)} done, {len(jobs)} to run, {a.workers} workers", flush=True)
    if jobs:
        # round-robin so every worker gets a mix of tasks (balances run time)
        shards = [jobs[k::a.workers] for k in range(a.workers)]
        import multiprocessing as mp
        ctx = mp.get_context("spawn")       # fresh process: own EGL + sim
        t0 = time.time()
        procs = [ctx.Process(target=_worker, args=(k, s, cfg, run_dir))
                 for k, s in enumerate(shards) if s]
        for pr in procs:
            pr.start()
        for pr in procs:
            pr.join()
        wall = time.time() - t0
        bad = [pr.exitcode for pr in procs if pr.exitcode != 0]
        if bad:
            print(f"WARNING: {len(bad)} worker(s) exited with codes {bad}; "
                  "re-run the same command to resume", flush=True)
        new = [r for r in load_rows(run_dir) if (r["task_index"], r["episode_index"]) not in done]
        print("\n=== this run ===")
        summarize(new, wall)
    print("\n=== all rows in", run_dir, "===")
    summarize(load_rows(run_dir))


if __name__ == "__main__":
    main()
