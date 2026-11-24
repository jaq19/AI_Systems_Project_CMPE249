"""Report figure: (a) safety vs work on obstacle_avoidance, (b) harm vs refusals on
reasoning_safety, with ordinary-task refusals at the same threshold for contrast.
Reads the trade-off CSVs written by compare.py. Needs matplotlib:

  /usr/bin/python3 -m pip install --user matplotlib
  /usr/bin/python3 analysis/figure.py            # writes docs/fig_tradeoff.pdf and .png
"""
import csv
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"
INK, MUTED, SURF = "#0b0b0b", "#52514e", "#fcfcfb"


def rows(path):
    return {r["arm"]: {k: float(v) for k, v in r.items() if k != "arm"} for r in csv.DictReader(open(path))}


def wilson(p, n, z=1.96):
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def err(p, n):
    lo, hi = wilson(p, n)
    return [[100 * (p - lo)], [100 * (hi - p)]]


oa = rows("docs/trade_off_oa.csv")
oa.update(rows("docs/trade_off_oa_stac.csv"))
rs = rows("docs/trade_off_rs3.csv")

plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "font.family": "DejaVu Sans"})
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})
for ax in (a, b):
    ax.set_facecolor(SURF)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.25, linewidth=.6, zorder=0)

# ---- (a) obstacle_avoidance: collisions vs task success
seq = ["watcher_oa_gentle", "watcher_oa_medium", "watcher_oa_strict"]
xs = [100 * oa[k]["collision"] for k in seq]; ys = [100 * oa[k]["success"] for k in seq]
a.plot(xs, ys, "-", color=BLUE, linewidth=2, zorder=2)
for k, lab in zip(seq, ["gentle", "medium", "strict"]):
    r = oa[k]
    a.errorbar(100 * r["collision"], 100 * r["success"], xerr=err(r["collision"], r["n"]), yerr=err(r["success"], r["n"]),
               fmt="o", color=BLUE, ms=8, mec=SURF, mew=2, elinewidth=.8, capsize=0, zorder=3, alpha=.9)
    a.annotate(f"{lab}\nrefused {100*r['refused']:.0f}%", (100 * r["collision"], 100 * r["success"]),
               textcoords="offset points", xytext=(8, -4), fontsize=8, color=INK, va="top")
c = oa["control_oa"]
a.errorbar(100 * c["collision"], 100 * c["success"], xerr=err(c["collision"], c["n"]), yerr=err(c["success"], c["n"]),
           fmt="s", color=GRAY, ms=9, mec=SURF, mew=2, elinewidth=.8, zorder=4)
a.annotate("no watcher", (100 * c["collision"], 100 * c["success"]), textcoords="offset points",
           xytext=(8, 6), fontsize=8, color=INK)
s = oa["watcher_oa_stac_medium"]
a.errorbar(100 * s["collision"], 100 * s["success"], xerr=err(s["collision"], s["n"]), yerr=err(s["success"], s["n"]),
           fmt="D", color=ORANGE, ms=7, mec=SURF, mew=2, elinewidth=.8, zorder=4)
a.annotate(f"STAC-only (exploratory)\nrefused {100*s['refused']:.1f}%", (100 * s["collision"], 100 * s["success"]),
           textcoords="offset points", xytext=(-10, 16), fontsize=8, color=INK, ha="right")
a.plot([], [], "-o", color=BLUE, label="watcher, frozen design (gentle→strict)")
a.plot([], [], "D", color=ORANGE, label="watcher, STAC-only")
a.plot([], [], "s", color=GRAY, label="no watcher")
a.set_xlabel("Robot collisions, % of episodes   ← safer")
a.set_ylabel("Tasks completed, % of episodes   ↑ more work done")
a.set_title("(a) Obstacle avoidance (n = 360 per arm, 95% CI)", fontsize=10, color=INK, loc="left")
a.legend(frameon=False, fontsize=8, loc="lower right")

# ---- (b) reasoning_safety: harm engaged, refusals on harmful vs ordinary instructions
labels = ["no watcher", "medium", "strict"]
harm = [rs["control_rs"]["collision"], rs["watcher_rs_medium"]["collision"], rs["watcher_rs_strict"]["collision"]]
ref_h = [0.0, rs["watcher_rs_medium"]["refused"], rs["watcher_rs_strict"]["refused"]]
ref_o = [0.0, oa["watcher_oa_medium"]["refused"], oa["watcher_oa_strict"]["refused"]]
w, gap = 0.26, 0.02
series = [("harm engaged (harmful instr.)", harm, BLUE, 132),
          ("refused: harmful instructions", ref_h, ORANGE, 132),
          ("refused: ordinary tasks (OA)", ref_o, AQUA, 360)]
for j, (name, vals, col, n) in enumerate(series):
    x = [i + (j - 1) * (w + gap) for i in range(3)]
    b.bar(x, [100 * v for v in vals], width=w, color=col, label=name, zorder=3)
    for xi, v in zip(x, vals):
        lo, hi = wilson(v, n)
        if v > 0:
            b.errorbar(xi, 100 * v, yerr=[[100 * (v - lo)], [100 * (hi - v)]], fmt="none", ecolor=INK, elinewidth=.8, zorder=4)
        b.text(xi, 100 * (hi if v > 0 else 0) + 1.5, f"{100*v:.0f}", ha="center", fontsize=7.5, color=INK)
b.set_xticks(range(3), labels)
b.set_ylim(0, 115)
b.set_yticks(range(0, 101, 20))
b.set_ylabel("% of episodes")
b.set_title("(b) Harmful instructions (n = 132 per arm)", fontsize=10, color=INK, loc="left")
b.legend(frameon=False, fontsize=8, loc="upper center", ncol=1, bbox_to_anchor=(0.5, 1.0))
b.set_ylim(0, 135)

fig.patch.set_facecolor(SURF)
plt.tight_layout()
plt.savefig("docs/fig_tradeoff.pdf", bbox_inches="tight", facecolor=SURF)
plt.savefig("docs/fig_tradeoff.png", dpi=200, bbox_inches="tight", facecolor=SURF)
print("wrote docs/fig_tradeoff.pdf and docs/fig_tradeoff.png")
