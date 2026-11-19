"""First-order direct effect of each late-layer neuron on log p(next token), using the original model's own
next-token distribution from pass 3 as the baseline (d_t = E[target] - E_p). Pass-2 activations + pass-3 hidden."""
import sys, pickle, torch, numpy as np; sys.path.insert(0, "sv"); sys.path.insert(0, "sv/runs/b10-test/work")
import harness as H, probe_lib as P
d2 = pickle.load(open("sv/runs/b10-test/work/out/p2.pkl", "rb"))
d3 = pickle.load(open("sv/runs/b10-test/work/out/p3_E1.pkl", "rb"))
its3 = d3["items"]; tok = H.tokenizer()
# map pass-2 texts to pass-3 items (same raw texts)
idx3 = {it["req"].get("text"): i for i, it in enumerate(its3) if it["kind"] == "lm"}
E = H.orig_tensor("model.embed_tokens.weight").to(H.DEVICE); gF = 1 + H.orig_tensor("model.norm.weight").to(H.DEVICE)
LAYERS = list(range(12, 18))
Vs = {}; Ws = {}
for l in LAYERS:
    g = 1 + H.orig_tensor(f"model.layers.{l}.post_feedforward_layernorm.weight").to(H.DEVICE)
    Ws[l] = H.orig_tensor(f"model.layers.{l}.mlp.down_proj.weight").to(H.DEVICE)
    Vs[l] = (gF * g)[:, None] * Ws[l]
acc = {}; dE1 = []
for i2, x in enumerate(d2["items"]):
    i3 = idx3[x["text"]]
    assert d2["tokens"][i2] == d3["tokens"][i3]
    ids = torch.tensor(tok.convert_tokens_to_ids(d2["tokens"][i2]), device=H.DEVICE)
    lp = P.logp(d3["hid"]["orig"][i3])[:-1]; lpe = P.logp(d3["hid"]["edit"][i3])[:-1]
    Ep = lp.exp() @ E
    dt = E[ids[1:]] - Ep  # [T-1, 640]
    obs = (lpe.gather(-1, ids[1:, None]) - lp.gather(-1, ids[1:, None]))[:, 0]  # observed change in log p(target), E1
    grp = x["lang"]
    for l in LAYERS:
        a = torch.tensor(d2["acts"][l][i2].astype(np.float32), device=H.DEVICE)
        rms = (a @ Ws[l].T).pow(2).mean(-1).add(1e-6).sqrt()
        c = a[:-1] * (dt @ Vs[l]) / rms[:-1, None]  # [T-1, 2048] first-order logp gain from the neuron (unscaled)
        s = acc.setdefault((grp, l), [0, 0]); s[0] = s[0] + c.sum(0); s[1] += c.shape[0]
        if l == 17:
            dE1.append((grp, float(-c[:, 1977].sum()), float(obs.sum()), c.shape[0]))
pickle.dump({k: (v[0].cpu(), v[1]) for k, v in acc.items()}, open("sv/runs/b10-test/work/out/an3.pkl", "wb"))
# check on E1: predicted (removal) vs observed change of sum log p(target), per group
from collections import defaultdict
agg = defaultdict(lambda: [0, 0, 0])
for g, pr, ob, n in dE1:
    agg[g][0] += pr; agg[g][1] += ob; agg[g][2] += n
for g, (pr, ob, n) in agg.items():
    print(f"E1 check {g:7s} predicted(unscaled) {pr / n:+.3f}/tok  observed {ob / n:+.4f} nats/tok")
