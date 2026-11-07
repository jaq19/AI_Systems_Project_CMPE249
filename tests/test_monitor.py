"""Unit tests for the watcher. No GPU, no LIBERO; needs only numpy.
Run:  python3 tests/test_monitor.py      (prints MONITOR OK)
"""
import sys
import numpy as np
sys.path.insert(0, ".")
from vla_monitor.consistency import ConsistencyMonitor, mmd2_rbf, PROCEED, PAUSE, REFUSE

H, K = 10, 5      # chunk horizon, executed steps per call (as pi05_libero)


class TrajPolicy:
    """Deterministic-plus-noise policy whose chunk at env step t describes the
    SAME absolute trajectory f(t), f(t+1), ... A correct overlap alignment sees
    it as consistent; a misaligned one does not. mode_flip: at each CALL the
    whole action distribution switches to one of two opposite trajectories
    (STAC's erratic case: conflicting action modes across time). A policy
    that is merely multimodal *within* every call has a stable distribution,
    and STAC correctly does not flag it (see the multimodal check below)."""

    def __init__(self, noise=0.005, mode_flip=False, seed=0):
        self.rng, self.noise, self.flip = np.random.default_rng(seed), noise, mode_flip
        self.t, self.last_chunk, self.sign = 0, None, 1
        self.within = False

    def f(self, t, sign):
        tt = np.arange(t, t + H)[:, None]
        a = np.zeros((H, 7))
        a[:, :3] = sign * 0.5 * np.sin(0.4 * tt + np.array([0.0, 1.0, 2.0]))
        a[:, 6] = -1.0
        return a

    def _one(self):
        sign = self.rng.choice([-1, 1]) if self.within else self.sign
        return self.f(self.t, sign) + self.rng.normal(0, self.noise, (H, 7))

    def act(self, obs):
        if self.flip:
            self.sign = -self.sign                     # distribution switches each call
        self.last_chunk = self._one()
        return self.last_chunk[:K]

    def sample(self, obs=None, n=4):
        return np.stack([self._one() for _ in range(n)])


def run(policy, monitor, calls=12, move=True, pause_every=None):
    monitor.reset("")
    out, step = [], 0
    for c in range(calls):
        policy.t = step
        chunk = policy.act({})
        obs = {"robot0_eef_pos": np.array([0.01 * step if move else 0.0, 0, 1.0])}
        d = monitor.step(obs, "", chunk, policy, step)
        out.append(d)
        if d == REFUSE:
            break
        if d == PROCEED and not (pause_every and c % pause_every == 0):
            step += K
    return out


def check(name, ok, detail=""):
    print(f"   [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        raise SystemExit("MONITOR FAILED")


print("=== MMD sanity ===")
rng = np.random.default_rng(0)
same = mmd2_rbf(rng.normal(0, 1, (8, 5)), rng.normal(0, 1, (8, 5)))
diff = mmd2_rbf(rng.normal(0, 1, (8, 5)), rng.normal(3, 1, (8, 5)))
check("shifted sets score higher than same-distribution sets", diff > same + 0.2,
      f"same={same:.3f} diff={diff:.3f}")

print("=== overlap alignment (the template's bug) ===")
m = ConsistencyMonitor(np.inf); run(TrajPolicy(), m)
aligned = float(np.mean(m.stac_hist[1:]))
# a monitor that ignores elapsed steps compares different moments:
class Misaligned(ConsistencyMonitor):
    def _stac(self, S, step):
        return super()._stac(S, self._prev_step if self._prev_step is not None else step)
m2 = Misaligned(np.inf); run(TrajPolicy(), m2)
mis = float(np.mean(m2.stac_hist[1:]))
check("consistent policy scores low when overlap is aligned", aligned < 0.15, f"aligned={aligned:.3f}")
check("misaligned comparison would score it higher", mis > aligned + 0.1, f"misaligned={mis:.3f}")

print("=== steady vs erratic ===")
m = ConsistencyMonitor(np.inf); run(TrajPolicy(), m); steady = np.mean(m.stac_hist[1:])
m = ConsistencyMonitor(np.inf); run(TrajPolicy(mode_flip=True), m); erratic = np.mean(m.stac_hist[1:])
check("erratic (mode switches between calls) scores higher", erratic > steady + 0.1,
      f"steady={steady:.3f} erratic={erratic:.3f}")
pm = TrajPolicy(); pm.within = True
m = ConsistencyMonitor(np.inf); run(pm, m); multi = np.mean(m.stac_hist[1:])
check("stably multimodal policy is NOT flagged (as in STAC)", multi < erratic,
      f"multimodal={multi:.3f}")

print("=== stuck vs moving ===")
m = ConsistencyMonitor(np.inf); run(TrajPolicy(), m, move=False); stuck = m.stuck_hist[-1]
m = ConsistencyMonitor(np.inf); run(TrajPolicy(), m, move=True); moving = m.stuck_hist[-1]
check("stuck arm scores ~1, moving arm ~0", stuck > 0.9 and moving < 0.2,
      f"stuck={stuck:.3f} moving={moving:.3f}")

print("=== decisions ===")
d = run(TrajPolicy(mode_flip=True), ConsistencyMonitor(np.inf))
check("threshold=inf never intervenes (calibration mode)", set(d) == {PROCEED})
d = run(TrajPolicy(mode_flip=True), ConsistencyMonitor(-1.0, refuse_after=3))
check("always-anomalous: PAUSE, PAUSE, then REFUSE (look again before refusing)",
      d == [PAUSE, PAUSE, REFUSE], str(d))
m = ConsistencyMonitor(0.5, refuse_after=3)
d = run(TrajPolicy(), m)
check("steady policy at a moderate threshold mostly proceeds",
      d.count(PROCEED) >= len(d) - 1, "".join(m.decisions))

print("=== paused calls don't count as 'stuck' ===")
m = ConsistencyMonitor(np.inf); run(TrajPolicy(), m, move=True, pause_every=2)
check("stuck stays low when the WATCHER held the arm", max(m.stuck_hist) < 0.5,
      f"max stuck={max(m.stuck_hist):.3f}")

print("=== DummyPolicy.sample leaves last_chunk alone ===")
from harness.policy import DummyPolicy
p = DummyPolicy(); p.act({}); lc = p.last_chunk.copy(); p.sample(n=3)
check("sample() does not overwrite last_chunk", np.array_equal(lc, p.last_chunk))
print("\nMONITOR OK")
