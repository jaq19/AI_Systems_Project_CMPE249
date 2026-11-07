"""Test-only stand-in for LIBERO, so the runner/launcher can be checked
anywhere (no MuJoCo, no GPU). Activated by VLA_FAKE_ENV=1 in run_slice.py,
or by calling install() directly. Never used for real results."""
import numpy as np


class _Task:
    def __init__(self, i):
        self.name, self.language, self.level = f"fake_task_{i}", f"do fake task {i}", i // 5


class _Env:
    """Hits a hazard at step 30 + task, succeeds at step 60 + 10*task if task is even."""

    def __init__(self, task_index):
        self.t, self.k = 0, task_index

    def step(self, action):
        self.t += 1
        hit = self.t == 30 + self.k
        succ = (self.k % 2 == 0) and self.t >= 60 + 10 * self.k
        info = {"vla_violation": hit, "vla_success": succ,
                "vla_pairs": {"robot~bottle_1": hit, "bowl_1~bottle_1": False},
                "vla_max_force": 3.0 if hit else 0.0}
        return {"agentview_image": np.zeros((8, 8, 3), np.uint8)}, 0.0, succ, info

    def hazard_summary(self):
        k = self.k % 2 == 1 and self.t >= 40
        return dict(knocked=k, first_knock_step=40 if k else -1,
                    knocked_hazards=["bottle_1"] if k else [],
                    hazard_max_disp_m={"bottle_1": 0.05 if k else 0.0},
                    hazard_max_tilt_deg={"bottle_1": 80.0 if k else 0.0})

    def close(self):
        pass


def install():
    from harness import adapter
    adapter.make_env = lambda suite, i, seed=0, img=256: (_Env(i), _Task(i))
    adapter.reset_env = lambda env, suite, i, ep: {"agentview_image": np.zeros((8, 8, 3), np.uint8)}
