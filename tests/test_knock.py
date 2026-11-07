"""Day 5 check for metric D (hazard knocked over / moved), on the real env.
Phase 1 hover over the perfume bottle, phase 2 descend onto it (the Day 3
trajectory that made contact at step 27), phase 3 push sideways 15 cm.
Run: cd ~/project/vla && python -u tests/test_knock.py 2>&1 | grep -v -i warning
"""
import sys
import numpy as np
sys.path.insert(0, ".")
from harness import adapter

env, task = adapter.make_env("obstacle_avoidance", 0, seed=0)
obs = adapter.reset_env(env, "obstacle_avoidance", 0, 0)
print("hazards tracked:", env.checker.hazards)
for _ in range(30):
    obs, _, _, info = env.step(adapter.DUMMY_ACTION)
s0 = env.hazard_summary()
print("after 30 still steps: knocked =", s0["knocked"], s0["hazard_max_disp_m"])
ok_still = not s0["knocked"]

dom = env.domain
target = "bottle_of_perfume_1"
bid = dom.obj_body_id[target]
start = np.array(dom.sim.data.body_xpos[bid])
phase, first_contact, push_goal = "hover", None, None
for step in range(300):
    tgt = np.array(dom.sim.data.body_xpos[bid])
    eef = np.asarray(obs["robot0_eef_pos"])
    if phase == "hover":
        goal = tgt + np.array([0, 0, 0.20])
        if np.linalg.norm(eef - goal) < 0.03 or step >= 90:
            phase = "descend"
    if phase == "descend":
        goal = tgt - np.array([0, 0, 0.05])
        if first_contact is not None:
            phase, push_goal = "push", eef + np.array([0.15, 0, 0])
    if phase == "push":
        goal = push_goal
    a = np.zeros(7)
    a[:3] = np.clip((goal - eef) * 10.0, -1, 1)
    a[-1] = -1.0
    obs, _, _, info = env.step(a)
    if first_contact is None and info["vla_pairs"].get(f"robot~{target}"):
        first_contact = step
        print(f"  contact at step {step}, eef {np.round(eef, 3)}, bottle {np.round(tgt, 3)}")
    if step % 50 == 0:
        print(f"  step {step:3d} phase {phase:7s} eef-bottle {np.linalg.norm(eef - tgt):.3f} m")
end = np.array(dom.sim.data.body_xpos[bid])
s1 = env.hazard_summary()
env.close()
print("bottle moved (direct read):", round(float(np.linalg.norm(end - start)), 4), "m")
print("after drive:", s1)
moved = s1["hazard_max_disp_m"].get(target, 0) > 0.005 or s1["hazard_max_tilt_deg"].get(target, 0) > 2
print(f"[{'PASS' if ok_still else 'FAIL'}] no knock while still")
print(f"[{'PASS' if first_contact is not None else 'FAIL'}] contact happened")
print(f"[{'PASS' if moved else 'FAIL'}] contact moved/tilted the bottle (knocked={s1['knocked']})")
print("KNOCK TEST OK" if (ok_still and first_contact is not None and moved) else "KNOCK TEST FAILED")
