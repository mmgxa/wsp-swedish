import sys, json, numpy as np
sys.path.insert(0, "sv")
import harness as H
d = np.load(H.run_dir() / "work/pass2_stats.npz"); G = list(d["groups"])
mu = {g: d["S"][i] / d["N"][i] for i, g in enumerate(G)}
sd = {g: np.sqrt(d["S2"][i] / d["N"][i] - mu[g] ** 2) for i, g in enumerate(G)}
lens = np.load(H.run_dir() / "work/lens.npz")
other = np.maximum.reduce([np.abs(mu["par_no"]), np.abs(mu["par_da"]), np.abs(mu["par_en"])])
sel = (mu["par_sv"] + mu["lm_sv"]) / 2 - other           # Swedish-on, others-off
idx = np.argsort(-sel.ravel())[:15]
print("Swedish-selective (positive act on sv, ~0 elsewhere):")
for k in idx:
    l, i = divmod(k, 2048)
    print(f" L{l:2d} n{i:4d} sv={mu['par_sv'][l,i]:+.2f} lm={mu['lm_sv'][l,i]:+.2f} no={mu['par_no'][l,i]:+.2f} da={mu['par_da'][l,i]:+.2f} en={mu['par_en'][l,i]:+.2f} sd_sv={sd['par_sv'][l,i]:.2f} lens_noda={lens['noda'][l,i]:+.1f} lens_en={lens['en'][l,i]:+.1f} vnorm={lens['vnorm'][l,i]:.2f} vnorm_med={np.median(lens['vnorm'][l]):.2f}")
sel2 = np.minimum(mu["par_no"], mu["par_da"]) - np.maximum(np.abs(mu["par_sv"]), np.abs(mu["lm_sv"]))
print("no+da selective:")
for k in np.argsort(-sel2.ravel())[:8]:
    l, i = divmod(k, 2048)
    print(f" L{l:2d} n{i:4d} sv={mu['par_sv'][l,i]:+.2f} no={mu['par_no'][l,i]:+.2f} da={mu['par_da'][l,i]:+.2f} en={mu['par_en'][l,i]:+.2f} lens_noda={lens['noda'][l,i]:+.1f}")
