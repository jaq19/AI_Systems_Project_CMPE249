import json, glob, statistics as st, sys
rows = [json.loads(l) for f in glob.glob(sys.argv[1] + "/part_w*.jsonl") for l in open(f) if l.strip()]
print(f"{'task':>4} {'n':>3} {'succ':>5} {'robot':>5} {'all':>5} {'knock':>5} {'knock_no_contact':>16} {'med_knock@':>10}  knocked hazards (median disp m / tilt deg)")
tot_nc = 0
for t in sorted({r["task_index"] for r in rows}):
    R = [r for r in rows if r["task_index"] == t]; n = len(R)
    p = lambda f: f"{100*sum(1 for r in R if f(r))/n:4.0f}%"
    nc = [r for r in R if r["knocked"] and r["first_violation_step"] < 0]; tot_nc += len(nc)
    ks = [r["first_knock_step"] for r in nc] or [r["first_knock_step"] for r in R if r["knocked"]]
    hz = {}
    for r in R:
        for h in r["knocked_hazards"]:
            hz.setdefault(h, []).append((r["hazard_max_disp_m"][h], r["hazard_max_tilt_deg"][h]))
    hs = ", ".join(f"{h.replace('_1','')}x{len(v)} ({st.median(a for a,_ in v):.2f}/{st.median(b for _,b in v):.0f})" for h, v in hz.items())
    print(f"{t:4d} {n:3d} {p(lambda r: r['success_raw'])} {p(lambda r: r['violation_robot'])} {p(lambda r: r['violation'])} "
          f"{p(lambda r: r['knocked'])} {len(nc):9d} ({100*len(nc)/n:3.0f}%) {(st.median(ks) if ks else -1):10.0f}  {hs}")
print(f"\nTOTAL n={len(rows)}  knocked-with-NO-registered-contact: {tot_nc} ({100*tot_nc/len(rows):.0f}%)")
