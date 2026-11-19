import sys, numpy as np, torch
from collections import Counter
sys.path.insert(0, "sv"); sys.path.insert(0, "sv/runs/b5-test/work")
import harness as H
from texts import PAR
tok = H.tokenizer()
cnt = {l: Counter() for l in ("sv", "no", "da", "en")}
for p in PAR:
    for l in cnt: cnt[l].update(H._ids(p[l]))
noda = cnt["no"] + cnt["da"]
T_noda = [t for t, c in noda.items() if c >= 2 and cnt["sv"][t] == 0]
T_sv = [t for t, c in cnt["sv"].items() if c >= 2 and noda[t] == 0]
T_en = [t for t, c in cnt["en"].items() if c >= 2 and cnt["sv"][t] == 0 and noda[t] == 0]
print(len(T_noda), len(T_sv), len(T_en))
print("sv:", [tok.decode([t]) for t in sorted(T_sv, key=lambda t: -cnt['sv'][t])[:30]])
E = H.orig_tensor("model.embed_tokens.weight")
gf = 1 + H.orig_tensor("model.norm.weight")
d = np.load(H.run_dir() / "work/pass2_stats.npz")
G = list(d["groups"]); mu = {g: d["S"][i] / d["N"][i] for i, g in enumerate(G)}
sd = np.sqrt(np.mean([d["S2"][G.index(g)] / d["N"][G.index(g)] - mu[g] ** 2 for g in ("par_sv", "par_no", "par_da", "par_en")], 0)) + 1e-6
dir_noda = E[T_noda].mean(0) - E[T_sv].mean(0)       # unembedding direction no/da minus sv
dir_en = E[T_en].mean(0) - E[T_sv].mean(0)
out = {}
for l in range(18):
    W = H.orig_tensor(f"model.layers.{l}.mlp.down_proj.weight")   # 640 x 2048
    gp = 1 + H.orig_tensor(f"model.layers.{l}.post_feedforward_layernorm.weight")
    V = (gp * gf)[:, None] * W                                # residual write, as seen by the final norm
    ln = (dir_noda @ V).numpy(); le = (dir_en @ V).numpy()
    out[l] = (ln, le, np.linalg.norm(V.numpy(), axis=0))
np.savez(H.run_dir() / "work/lens.npz", noda=np.stack([out[l][0] for l in range(18)]), en=np.stack([out[l][1] for l in range(18)]),
         vnorm=np.stack([out[l][2] for l in range(18)]))
# DLA of each neuron in Swedish contexts toward no/da (mean act * lens)
for g in ("par_sv", "par_no", "par_da", "gen_sv", "lm_sv"):
    dla = np.stack([mu[g][l] * out[l][0] for l in range(18)])
    print(g, "per-layer summed DLA toward no/da:", " ".join(f"{x:+.2f}" for x in dla.sum(1)))
dla_sv = np.stack([mu["par_sv"][l] * out[l][0] for l in range(18)])
dla_noda = np.stack([(mu["par_no"][l] + mu["par_da"][l]) / 2 * out[l][0] for l in range(18)])
z = ((mu["par_no"] + mu["par_da"]) / 2 - mu["par_sv"]) / sd
idx = np.argsort(-(dla_noda - dla_sv).ravel())[:25]
print("top neurons by (DLA in no/da text - DLA in sv text) toward no/da:")
for k in idx:
    l, i = divmod(k, 2048)
    print(f" L{l:2d} n{i:4d} z={z[l,i]:+.2f} mu_sv={mu['par_sv'][l,i]:+.3f} mu_no={mu['par_no'][l,i]:+.3f} mu_da={mu['par_da'][l,i]:+.3f} mu_gensv={mu['gen_sv'][l,i]:+.3f} lens={out[l][0][i]:+.3f} dla_sv={dla_sv[l,i]:+.3f} dla_noda={dla_noda[l,i]:+.3f}")
