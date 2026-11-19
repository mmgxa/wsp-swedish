import sys, json, numpy as np
sys.path.insert(0, "sv")
import harness as H
d = np.load(H.run_dir() / "work/pass2_stats.npz"); G = list(d["groups"])
mu = {g: d["S"][i] / d["N"][i] for i, g in enumerate(G)}
lens = np.load(H.run_dir() / "work/lens.npz")
M = d["item_mean"]; meta = json.load(open(H.run_dir() / "work/pass2_meta.json"))
drift = [k for k, m in enumerate(meta) if m["grp"] in ("gen_no", "gen_da") or m["i"] in ("gen-29", "own-21", "own-16")]
print("drift transcripts:", [meta[k]["i"] for k in drift])
c = []
for l in range(1, 18):
    for i in range(2048):
        on = max(mu["par_no"][l, i], mu["par_da"][l, i])
        off = max(abs(mu["par_sv"][l, i]), abs(mu["lm_sv"][l, i]), abs(mu["par_en"][l, i]), abs(mu["gen_sv"][l, i]))
        if on >= 0.3 and off <= 0.12 and lens["noda"][l, i] > 5:
            c.append((on * lens["noda"][l, i], l, i, on, off))
for s, l, i, on, off in sorted(c, reverse=True)[:12]:
    print(f"L{l:2d} n{i:4d} no={mu['par_no'][l,i]:+.2f} da={mu['par_da'][l,i]:+.2f} sv={mu['par_sv'][l,i]:+.2f} lm={mu['lm_sv'][l,i]:+.2f} gensv={mu['gen_sv'][l,i]:+.2f} en={mu['par_en'][l,i]:+.2f} lens_noda={lens['noda'][l,i]:+.1f} lens_en={lens['en'][l,i]:+.1f} drift_items_mean={M[drift, l, i].mean():+.2f}")
