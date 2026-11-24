"""Does the watcher's score rise BEFORE the first robot collision?
Pure standard library (Mac). Usage:

  python3 analysis/precursor.py logs/watcher_oa_gentle logs/watcher_oa_medium logs/watcher_oa_strict

Each watcher look happens at env step s (steps executed so far). A '.' look
executes a 5-action chunk; 'p' re-looks at the same s. We keep the FIRST look
at each s. A look is a POSITIVE if the first robot collision c falls within
the next H env steps (s < c <= s + H). NEGATIVES are looks from episodes with
no robot collision, restricted to the step range the positives occupy, so
that "late in the episode" cannot masquerade as "about to collide".

Prints:
  1. AUROC of score / STAC / stuck for H = 5, 25, 50 (0.5 = no warning signal),
     with a bootstrap 95% CI over episodes
  2. mean score aligned to the collision step (-60 .. +60 steps)
  3. how often the watcher flagged (score > threshold) before the collision
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
from metrics import load  # noqa: E402

CHUNK = 5


def looks(r):
    """[(env_step, score, stac, stuck)] with the first look at each env step."""
    out, s, seen = [], 0, set()
    sc, st, sk, dec = r.get("monitor_scores", []), r.get("monitor_stac", []), r.get("monitor_stuck", []), r.get("monitor_decisions", "")
    for i, d in enumerate(dec):
        if i >= len(sc):
            break
        if s not in seen:
            seen.add(s)
            out.append((s, sc[i], st[i] if i < len(st) else 0.0, sk[i] if i < len(sk) else 0.0))
        if d == ".":
            s += CHUNK
    return out


def auroc(pos, neg):
    """Mann-Whitney AUROC with ties counted as 1/2."""
    if not pos or not neg:
        return float("nan")
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    rank_sum, i, n = 0.0, 0, len(allv)
    while i < n:
        j = i
        while j < n and allv[j][0] == allv[i][0]:
            j += 1
        avg = (i + 1 + j) / 2.0
        rank_sum += avg * sum(1 for k in range(i, j) if allv[k][1] == 1)
        i = j
    npos, nneg = len(pos), len(neg)
    return (rank_sum - npos * (npos + 1) / 2) / (npos * nneg)


def split(eps, H, comp):
    pos_eps, neg_eps = [], []
    for r, L in eps:
        c = r["first_violation_robot_step"]
        if c >= 0:
            pos_eps.append([(x[0], x[comp]) for x in L if x[0] < c <= x[0] + H])
        else:
            neg_eps.append([(x[0], x[comp]) for x in L])
    pos_eps = [p for p in pos_eps if p]
    steps = sorted(s for p in pos_eps for s, _ in p)
    if not steps:
        return pos_eps, [], (0, 0)
    lo, hi = steps[len(steps) // 10], steps[(9 * len(steps)) // 10]
    neg_eps = [[(s, v) for s, v in n if lo <= s <= hi] for n in neg_eps]
    return pos_eps, [n for n in neg_eps if n], (lo, hi)


def boot(pos_eps, neg_eps, B=200, seed=0):
    rng = random.Random(seed)
    vals = []
    for _ in range(B):
        P = [v for e in (rng.choice(pos_eps) for _ in pos_eps) for _, v in e]
        N = [v for e in (rng.choice(neg_eps) for _ in neg_eps) for _, v in e]
        vals.append(auroc(P, N))
    vals.sort()
    return vals[int(0.025 * B)], vals[int(0.975 * B) - 1]


def main(dirs):
    eps, by_arm = [], []
    for d in dirs:
        cp = os.path.join(d, "config.json")
        cfg = json.load(open(cp)) if os.path.exists(cp) else {}
        E = [(r, looks(r)) for r in load(d)]
        E = [(r, L) for r, L in E if L]
        eps += E
        by_arm.append((os.path.basename(d.rstrip("/")), cfg.get("threshold"), E))
    ncol = sum(1 for r, _ in eps if r["first_violation_robot_step"] >= 0)
    print(f"episodes with watcher scores: {len(eps)}   with a robot collision: {ncol}")

    print("\n" + "=" * 90 + "\n1. AUROC: does the score separate 'collision within H steps' from collision-free looks?\n" + "=" * 90)
    print("   (0.5 = no signal; > 0.5 = score higher before collisions; < 0.5 = LOWER before collisions)")
    for H in (5, 25, 50):
        for comp, name in ((1, "score"), (2, "STAC"), (3, "stuck")):
            pe, ne, (lo, hi) = split(eps, H, comp)
            a = auroc([v for e in pe for _, v in e], [v for e in ne for _, v in e])
            ci = boot(pe, ne) if pe and ne else (float("nan"),) * 2
            print(f"   H={H:3d}  {name:6s} AUROC {a:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]   "
                  f"pos looks {sum(map(len, pe)):5d} ({len(pe)} eps)   neg looks {sum(map(len, ne)):6d} ({len(ne)} eps)   steps {lo}-{hi}")

    print("\n" + "=" * 90 + "\n2. MEAN SCORE ALIGNED TO THE FIRST ROBOT COLLISION (step 0 = collision)\n" + "=" * 90)
    bins = list(range(-60, 61, 20))
    ref = [v for r, L in eps if r["first_violation_robot_step"] < 0 for s, v, _, _ in L]
    print(f"   collision-free episodes, all looks: mean score {sum(ref)/max(len(ref),1):.3f}")
    for comp, name in ((1, "score"), (2, "STAC"), (3, "stuck")):
        row = []
        for b in bins:
            v = [x[comp] for r, L in eps if r["first_violation_robot_step"] >= 0
                 for x in L if b - 10 <= x[0] - r["first_violation_robot_step"] < b + 10]
            row.append(f"{b:+4d}:{(sum(v)/len(v) if v else float('nan')):.3f}")
        print(f"   {name:6s} " + "  ".join(row))

    print("\n" + "=" * 90 + "\n3. DID THE WATCHER FLAG BEFORE THE COLLISION?  (score > its arm's threshold)\n" + "=" * 90)
    for an, t, E in by_arm:
        if t is None or t != t or not E:
            continue
        col = [(r, L) for r, L in E if r["first_violation_robot_step"] >= 0]
        free = [(r, L) for r, L in E if r["first_violation_robot_step"] < 0]
        cs = sorted(r["first_violation_robot_step"] for r, _ in col)
        med = cs[len(cs) // 2] if cs else 0
        f_col = sum(1 for r, L in col if any(v > t for s, v, _, _ in L if s < r["first_violation_robot_step"]))
        f_free = sum(1 for r, L in free if any(v > t for s, v, _, _ in L if s < med))
        print(f"   {an:20s} thr={t:.3f}  flagged before collision: {f_col}/{len(col)} ({100*f_col/max(len(col),1):.0f}%)"
              f"   collision-free eps flagged before step {med}: {f_free}/{len(free)} ({100*f_free/max(len(free),1):.0f}%)")
    print("   (if the two percentages are similar, a flag says nothing about an upcoming collision)")


if __name__ == "__main__":
    main(sys.argv[1:])
