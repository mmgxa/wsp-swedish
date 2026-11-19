import sys, json, numpy as np
sys.path.insert(0, "sv")
import harness as H
d = np.load(H.run_dir() / "work/pass2_stats.npz")
G = list(d["groups"]); S, S2, N = d["S"], d["S2"], d["N"]
mu = {g: S[i] / N[i] for i, g in enumerate(G)}
var = {g: S2[i] / N[i] - mu[g] ** 2 for i, g in enumerate(G)}
sd = np.sqrt(np.mean([var[g] for g in ("par_sv", "par_no", "par_da", "par_en")], 0)) + 1e-6
z_noda = ((mu["par_no"] + mu["par_da"]) / 2 - mu["par_sv"]) / sd
z_en = (mu["par_en"] - mu["par_sv"]) / sd
print("layer  #|z_noda|>1  max z_noda  min z_noda  #|z_en|>1  meanabs act sv")
for l in range(18):
    print(l, (abs(z_noda[l]) > 1).sum(), round(z_noda[l].max(), 2), round(z_noda[l].min(), 2), (abs(z_en[l]) > 1).sum(), round(np.abs(mu["par_sv"][l]).mean(), 3))
np.save(H.run_dir() / "work/z_noda.npy", z_noda)
