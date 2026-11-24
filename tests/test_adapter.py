"""Day 2 acceptance test for harness/adapter.py.

Drives the arm straight into a registered hazard with a scripted controller
(no policy needed). Checks that our violation signal fires, and records what
the benchmark's own info['cost'] said at the same moment.

Run:  cd ~/project/vla && python tests/test_adapter.py 2>&1 | tee docs/test_adapter.txt
"""
import sys, time
import numpy as np
sys.path.insert(0, ".")
from harness import adapter

RESULTS = {}


def drive_into(env, obs, target_name, label, max_steps=220):
    """Hover above target, then descend into it. Return (step, info) at first contact."""
    dom = env.domain
    descending = False
    for step in range(max_steps):
        tgt = np.array(dom.sim.data.body_xpos[dom.obj_body_id[target_name]])
        eef = np.asarray(obs["robot0_eef_pos"])
        hover = tgt + np.array([0, 0, 0.20])
        if not descending and (np.linalg.norm(eef - hover) < 0.03 or step >= 90):
            descending = True
        goal = (tgt - np.array([0, 0, 0.05])) if descending else hover
        a = np.zeros(7)
        a[:3] = np.clip((goal - eef) * 10.0, -1, 1)
        a[-1] = -1.0
        obs, reward, done, info = env.step(a)
        if step % 20 == 0:
            print(f"     step {step:3d}  dist={np.linalg.norm(goal - eef):.3f}  "
                  f"ours={info['vla_pairs'].get(label)}  shipped={info['shipped_cost']}")
        if info["vla_pairs"].get(label):
            return step, info, obs
    return None, info, obs


def check(name, ok, detail=""):
    RESULTS[name] = ok
    print(f"   [{'PASS' if ok else 'FAIL'}] {name} {detail}")


# -------------------------------------------------------------------------
print("=== 1. Task lists ===")
for suite in ("obstacle_avoidance", "reasoning_safety"):
    tasks = adapter.list_tasks(suite)
    print(f"   {suite}: {len(tasks)} tasks")
    for i, n, l in tasks:
        lv, lid = adapter.task_level(suite, i)
        print(f"     [{i:2d}] L{lv}.{lid}  {l[:70]}")
print("   calibration tasks:", adapter.CALIB_TASKS, " report tasks:", adapter.REPORT_TASKS)

# -------------------------------------------------------------------------
print("\n=== 2. obstacle_avoidance task 0: registered pairs ===")
SUITE = "obstacle_avoidance"
env, task = adapter.make_env(SUITE, 0, seed=0)
obs = adapter.reset_env(env, SUITE, 0, 0)
ck = env.checker
print("   robot geoms:", len(ck.robot_geoms))
for label, A, B in ck.pairs:
    print(f"   pair {label:55s} geoms {len(A)} x {len(B)}")
raw = env.domain.parsed_problem["constraints"]
print(f"   BDDL declares {len(raw)} constraints; info['cost'] can hold at most "
      f"{len({c[0] for c in raw})} of them (keyed by predicate name)")
img = adapter.get_image(obs)
print("   image", img.shape, img.dtype, img.min(), "-", img.max())
check("image renders", img.max() > 0)

print("\n   no-op step:")
obs, r, d, info = env.step(np.r_[np.zeros(6), -1.0])
print("   info keys:", sorted(info.keys()))
check("no violation at rest", not info["vla_violation"], f"pairs={info['vla_pairs']}")

robot_pair = next(l for l, _, _ in ck.pairs if l.startswith("robot~"))
hazard = robot_pair.split("~", 1)[1]
print(f"\n   driving the gripper into {hazard} ...")
step, info, obs = drive_into(env, obs, hazard, robot_pair)
check("our signal fires on robot-hazard contact", step is not None, f"(step {step})")
if step is not None:
    print("   at contact: max normal force", round(info["vla_max_force"], 2), "N")
    print("   at contact: shipped info['cost'] =", info["shipped_cost"])
    builtin = env.domain.check_robot_contact(env.domain.get_object(hazard))
    print("   at contact: benchmark's own check_robot_contact() =", builtin)
    RESULTS["_shipped_robot_contact"] = bool(builtin)
env.close()

# -------------------------------------------------------------------------
print("\n=== 3. reasoning_safety task 0: goal-free task ===")
SUITE = "reasoning_safety"
env, task = adapter.make_env(SUITE, 0, seed=0)
obs = adapter.reset_env(env, SUITE, 0, 0)
print("   instruction:", task.language)
print("   obj_of_interest:", env.checker.interest, " has_goal:", env.checker.has_goal)
obs, r, d, info = env.step(np.r_[np.zeros(6), -1.0])
inner_success = env.domain._check_success()
print(f"   benchmark _check_success() after 1 no-op step: {inner_success}")
check("wrapper does not end goal-free episode at step 1", d is False)
check("wrapper reports no success", info["vla_success"] is False)
target = env.checker.interest[0]
print(f"\n   driving the gripper into {target} ...")
step, info, obs = drive_into(env, obs, target, f"robot~{target}")
check("harm_engaged fires on touching the object", step is not None and info["vla_violation"],
      f"(step {step})")
env.close()

# -------------------------------------------------------------------------
print("\n=== 4. Every task builds (catches path bugs, e.g. reasoning task 11) ===")
for suite in ("obstacle_avoidance", "reasoning_safety"):
    bad = []
    for i in range(15):
        try:
            e, _ = adapter.make_env(suite, i, seed=0)
            adapter.reset_env(e, suite, i, 0)
            e.close()
        except Exception as ex:
            bad.append((i, repr(ex)[:120]))
    check(f"{suite}: all 15 tasks build", not bad, str(bad))

# -------------------------------------------------------------------------
print("\n=== 5. Repeatable ===")
for suite in ("obstacle_avoidance", "reasoning_safety"):
    pics = []
    for ep in (0, 0, 1):
        e, _ = adapter.make_env(suite, 0, seed=0)
        pics.append(adapter.get_image(adapter.reset_env(e, suite, 0, ep)).astype(float))
        e.close()
    check(f"{suite}: same seed+episode -> same picture", np.allclose(pics[0], pics[1]))
    print(f"   {suite}: episode 1 differs from episode 0: {not np.allclose(pics[0], pics[2])}")

# -------------------------------------------------------------------------
print("\n=== SUMMARY ===")
fails = [k for k, v in RESULTS.items() if not k.startswith("_") and not v]
if "_shipped_robot_contact" in RESULTS:
    print("   benchmark's own robot-contact check at the moment of contact:",
          RESULTS["_shipped_robot_contact"],
          "(expected False = shipped signal is broken, as diagnosed)")
print("   ADAPTER OK" if not fails else f"   FAILED: {fails}")
sys.exit(1 if fails else 0)
