import sys, json, numpy as np
sys.path.insert(0, "sv")
import harness as H
d = np.load(H.run_dir() / "work/pass2_stats.npz"); M = d["item_mean"]
meta = json.load(open(H.run_dir() / "work/pass2_meta.json"))
neur = [(14, 1542), (13, 72), (17, 700), (17, 1418), (15, 1944), (15, 459), (14, 1594), (16, 570)]
grp = np.array([m["grp"] for m in meta])
for g in sorted(set(grp)):
    k = grp == g
    print(f"{g:10s} n={k.sum():3d} " + " ".join(f"L{l}n{i}={M[k, l, i].mean():+.2f}" for l, i in neur))
print("per gen item (non-sv labels and some sv):")
for k, m in enumerate(meta):
    if m["grp"].startswith("gen_") and (m["grp"] != "gen_sv" or k % 9 == 0):
        print(f"  {m['i']:8s} {m['grp']:9s} " + " ".join(f"{M[k, l, i]:+.2f}" for l, i in neur[:4]))
