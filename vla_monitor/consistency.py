"""The watcher: consistency (STAC-lite) + progress (stuck detector).

Says PROCEED, PAUSE (wait, re-sample, look again) or REFUSE, a few times a
second, from what the policy is about to do. It never drives and never sees
the simulator's contact data.

Consistency follows Agia et al., "Unpacking Failure Modes of Generative
Policies" (CoRL 2024, arXiv 2410.04640), STAC: at each policy call, draw B
action chunks from the (stochastic) policy and compare their distribution over
the time window that OVERLAPS the previous call's chunks, with MMD^2 under an
RBF kernel (median-heuristic bandwidth, so it is scale-free). Differences from
the paper, for compute: B=4 samples (paper: up to 256), and a stuck detector
stands in for their VLM progress monitor.

Fixes vs the guide's template (2026-11-07):
  * compares the overlapping timesteps (prev[k:] vs cur[:H-k]); the template
    compared chunk t to chunk t-1 index by index, i.e. different moments
  * uses several samples per call (distribution consistency, as in STAC);
    the template used one, so policy noise looked like inconsistency
  * excludes the gripper dimension (near-binary; would dominate distances)
  * REFUSE only after `refuse_after` consecutive anomalous looks ("wait and
    look again" first); the template refused on a single score > 1.5 x thr
  * stuck detector skips paused calls (the arm did not move because WE held it)
"""
import math
import numpy as np

PROCEED, PAUSE, REFUSE = "PROCEED", "PAUSE", "REFUSE"


def mmd2_rbf(X, Y):
    """Unbiased-ish MMD^2 between sample sets X (n,d) and Y (m,d), RBF kernel,
    bandwidth = median pairwise distance of the pooled set. Range ~[0, 2]."""
    Z = np.concatenate([X, Y], 0)
    d2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    med = np.median(d2[np.triu_indices(len(Z), 1)])
    gamma = 1.0 / (med + 1e-8)
    K = np.exp(-gamma * d2)
    n = len(X)
    kxx, kyy, kxy = K[:n, :n], K[n:, n:], K[:n, n:]
    off = lambda A: (A.sum() - np.trace(A)) / max(A.shape[0] * (A.shape[0] - 1), 1)
    return float(max(off(kxx) + off(kyy) - 2 * kxy.mean(), 0.0))


class ConsistencyMonitor:
    name = "stac_progress"

    def __init__(self, threshold, n_samples=4, weight=0.6, stuck_window=4,
                 stuck_scale=0.02, refuse_after=3, dims=slice(0, 6)):
        self.threshold = float(threshold)
        self.n_samples, self.weight = int(n_samples), float(weight)
        self.stuck_window, self.stuck_scale = int(stuck_window), float(stuck_scale)
        self.refuse_after, self.dims = int(refuse_after), dims
        self.reset("")

    def params(self):
        return dict(name=self.name, threshold=self.threshold, n_samples=self.n_samples,
                    weight=self.weight, stuck_window=self.stuck_window,
                    stuck_scale=self.stuck_scale, refuse_after=self.refuse_after)

    def reset(self, instruction):
        self.instruction = instruction
        self.history, self.stac_hist, self.stuck_hist, self.decisions = [], [], [], []
        self._prev, self._prev_step = None, None
        self._positions, self._consecutive = [], 0

    # ---- components ---------------------------------------------------------
    def _samples(self, chunk, policy):
        first = getattr(policy, "last_chunk", None)
        first = np.asarray(first if first is not None else chunk, dtype=float)
        S = [first]
        if self.n_samples > 1 and hasattr(policy, "sample"):
            extra = np.asarray(policy.sample(n=self.n_samples - 1), dtype=float)
            S += list(extra.reshape((-1,) + first.shape))
        H = min(s.shape[0] for s in S)
        return np.stack([s[:H, self.dims] for s in S])          # (B, H, d)

    def _stac(self, S, step):
        prev, prev_step = self._prev, self._prev_step
        self._prev, self._prev_step = S, step
        if prev is None:
            return 0.0
        k = max(int(step - prev_step), 0)                      # env steps executed since
        H = min(prev.shape[1], S.shape[1])
        if k >= H:
            return 0.0
        X = prev[:, k:H].reshape(len(prev), -1)                # same future moments
        Y = S[:, :H - k].reshape(len(S), -1)
        return mmd2_rbf(X, Y) / 2.0                            # -> [0, 1]

    def _stuck(self, obs, step):
        pos = obs.get("robot0_eef_pos") if isinstance(obs, dict) else None
        if pos is None:
            return 0.0
        if self._positions and self._positions[-1][0] == step:
            pass                                               # paused: no new motion data
        else:
            self._positions.append((step, np.asarray(pos, dtype=float)))
            self._positions = self._positions[-(self.stuck_window + 1):]
        if len(self._positions) <= self.stuck_window:
            return 0.0
        moved = float(np.linalg.norm(self._positions[-1][1] - self._positions[0][1]))
        return math.exp(-moved / self.stuck_scale)

    # ---- interface used by harness/runner.py --------------------------------
    def score(self, obs, instruction, chunk, policy, step):
        stac = self._stac(self._samples(chunk, policy), step)
        stuck = self._stuck(obs, step)
        self.stac_hist.append(round(stac, 5))
        self.stuck_hist.append(round(stuck, 5))
        return self.weight * stac + (1 - self.weight) * stuck

    def step(self, obs, instruction, chunk, policy, step):
        s = self.score(obs, instruction, chunk, policy, step)
        self.history.append(round(float(s), 5))
        if s > self.threshold:
            self._consecutive += 1
            d = REFUSE if self._consecutive >= self.refuse_after else PAUSE
        else:
            self._consecutive = 0
            d = PROCEED
        self.decisions.append({PROCEED: ".", PAUSE: "p", REFUSE: "R"}[d])
        return d
