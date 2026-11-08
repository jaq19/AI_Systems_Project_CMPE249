"""Robustness check for precursor.py: AUROC within 25-step bins of the episode,
so positives and negatives are compared at the SAME point in the episode.
Weighted by the number of positive looks per bin.

  python3 analysis/precursor_strat.py logs/watcher_oa_gentle logs/watcher_oa_medium logs/watcher_oa_strict
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from metrics import load  # noqa: E402
from precursor import auroc, looks, split  # noqa: E402

BIN = 25


def main(dirs):
    eps = [(r, L) for d in dirs for r in load(d) for L in [looks(r)] if L]
    for H in (5, 25):
        print(f"\nH={H}: per-bin AUROC (score / STAC / stuck), positives = looks within {H} steps of the first collision")
        tot = {1: [0.0, 0], 2: [0.0, 0], 3: [0.0, 0]}
        rows = {}
        for comp in (1, 2, 3):
            pe, ne, _ = split(eps, H, comp)
            P = [(s, v) for e in pe for s, v in e]
            N = [(s, v) for e in ne for s, v in e]
            for b in sorted({s // BIN for s, _ in P}):
                p = [v for s, v in P if s // BIN == b]
                n = [v for s, v in N if s // BIN == b]
                if len(p) < 10 or len(n) < 10:
                    continue
                a = auroc(p, n)
                rows.setdefault(b, {})[comp] = (a, len(p), len(n))
                tot[comp][0] += a * len(p); tot[comp][1] += len(p)
        for b in sorted(rows):
            cells = "   ".join(f"{rows[b][c][0]:.3f}" if c in rows[b] else "  -  " for c in (1, 2, 3))
            any_c = next(iter(rows[b].values()))
            print(f"   steps {b*BIN:3d}-{b*BIN+BIN-1:3d}   {cells}   (pos {any_c[1]:4d}, neg {any_c[2]:5d})")
        print("   weighted   " + "   ".join(f"{tot[c][0]/max(tot[c][1],1):.3f}" for c in (1, 2, 3)) + "   <- step-matched AUROC")


if __name__ == "__main__":
    main(sys.argv[1:])
