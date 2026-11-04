"""Day 3 acceptance test: does pi0.5 turn an observation into a sensible action?

Needs the server running:  cd ~/project/openpi && uv run scripts/serve_policy.py --env LIBERO
Run:  cd ~/project/vla && python tests/test_policy.py 2>&1 | tee docs/test_policy.txt

Section 3 runs obstacle_avoidance tasks with violations LOGGED, not ending the
episode. That separates "can the policy do the task at all" (the wiring check)
from "does it collide" (the experiment). A correctly wired pi05_libero should
complete some of these, since the base tasks come from LIBERO-10, where
openpi reports 92.4%.
"""
import os, sys, time
import numpy as np
sys.path.insert(0, ".")
from harness import adapter
from harness.policy import Pi05Policy

SUITE = "obstacle_avoidance"
MAX_STEPS = 520           # openpi's libero_10 horizon (longest demo: 505)
os.makedirs("docs/videos", exist_ok=True)

print("=== 1. Controller actually in use ===")
env, task = adapter.make_env(SUITE, 0, seed=0)
cs = adapter.controller_summary(env)
print("  ", cs)
assert abs(cs["output_max"][0] - 0.05) < 1e-6, "controller override did not apply"
env.close()

print("\n=== 2. Server, one inference ===")
pol = Pi05Policy()
print("   server metadata:", pol.metadata)
env, task = adapter.make_env(SUITE, 0, seed=0)
obs = adapter.reset_env(env, SUITE, 0, 0)
pol.reset(task.language)
t = time.perf_counter(); a = pol.act(obs); dt = time.perf_counter() - t
print(f"   first call {dt:.2f}s (includes JIT warm-up), chunk {pol.last_chunk.shape}, executing {a.shape}")
t = time.perf_counter(); pol.act(obs); print(f"   warm call {time.perf_counter() - t:.3f}s")
print("   first action:", np.round(a[0], 3))
s = pol.sample(n=4)
print("   spread across 4 samples of the same obs (mean std of xyz):",
      round(float(s[:, :, :3].std(0).mean()), 4))
env.close()


def episode(task_index, ep, video=False):
    env, task = adapter.make_env(SUITE, task_index, seed=0)
    obs = adapter.reset_env(env, SUITE, task_index, ep)
    pol.reset(task.language)
    frames, first_v, pairs_hit, success, steps = [], -1, set(), False, 0
    while steps < MAX_STEPS and not success:
        for act in pol.act(obs):
            obs, r, done, info = env.step(act)
            steps += 1
            if video:
                frames.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
            if info["vla_violation"] and first_v < 0:
                first_v = steps
            pairs_hit |= {k for k, v in info["vla_pairs"].items() if v}
            if info["vla_success"]:
                success = True
                break
    env.close()
    if video and frames:
        try:
            import imageio
            path = f"docs/videos/oa_t{task_index}_ep{ep}.mp4"
            imageio.mimwrite(path, frames, fps=20)
            print("     video:", path)
        except Exception as e:
            print("     (no video:", e, ")")
    return dict(task=task_index, ep=ep, success=success, steps=steps,
                first_violation=first_v, pairs=sorted(pairs_hit))


print("\n=== 3. Wiring check: 2 episodes on each L0 task, violations logged only ===")
rows = []
t0 = time.time()
for ti in range(5):
    for ep in range(2):
        r = episode(ti, ep, video=(ep == 0))
        rows.append(r)
        print(f"   task {ti} ep {ep}: success={r['success']!s:5} steps={r['steps']:3d} "
              f"first_violation={r['first_violation']:4d} pairs={r['pairs']}")
el = time.time() - t0
n = len(rows)
sr = sum(r["success"] for r in rows) / n
vr = sum(r["first_violation"] >= 0 for r in rows) / n
print(f"\n   success (ignoring collisions) {sr:.0%}   any violation {vr:.0%}   "
      f"{el / n:.0f}s/episode   {pol.infer_seconds / max(pol.calls, 1) * 1000:.0f} ms/inference")

print("\n=== SUMMARY ===")
if sr == 0:
    print("   0 successes: watch docs/videos/*.mp4. If the arm moves purposefully but misses,")
    print("   it may be genuine; if it drifts or flails, the wiring is wrong (image flip,")
    print("   state order, controller gains). Paste this output to Claude either way.")
else:
    print("   POLICY OK: pi0.5 turns observations into actions that complete tasks.")
if vr == 0:
    print("   No violation fired in any episode. Don't trust zeros: run tests/test_adapter.py.")
