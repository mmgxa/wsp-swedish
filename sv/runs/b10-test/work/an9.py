"""Like an4.py, but the first-order gain of removing a neuron is split into a temperature part (change along the
centred logits, under the model's own distribution) and the remainder (ranking change). Uses pass-2 acts and pass-3
orig hidden states. Output: per group/layer sums of total gain and of temperature-free gain."""
import sys, pickle, torch, numpy as np; sys.path.insert(0, "sv"); sys.path.insert(0, "sv/runs/b10-test/work")
import harness as H, probe_lib as P
d2 = pickle.load(open("sv/runs/b10-test/work/out/p2.pkl", "rb"))
d3 = pickle.load(open("sv/runs/b10-test/work/out/p3_E1.pkl", "rb"))
its3 = d3["items"]; tok = H.tokenizer()
idx3 = {it["req"].get("text"): i for i, it in enumerate(its3) if it["kind"] == "lm"}
dev = H.dev_items("lm", 30); devgrp = {x["text"]: ("svdevA" if k < 15 else "svdevB") for k, x in enumerate(dev)}
E = H.orig_tensor("model.embed_tokens.weight").to(H.DEVICE); gF = 1 + H.orig_tensor("model.norm.weight").to(H.DEVICE)
LAYERS = list(range(12, 18))
W, gp = {}, {}
for l in LAYERS:
    gp[l] = 1 + H.orig_tensor(f"model.layers.{l}.post_feedforward_layernorm.weight").to(H.DEVICE)
    W[l] = H.orig_tensor(f"model.layers.{l}.mlp.down_proj.weight").to(H.DEVICE)
acc = {}
for i2, x in enumerate(d2["items"]):
    i3 = idx3[x["text"]]; grp = devgrp.get(x["text"], x["lang"])
    ids = torch.tensor(tok.convert_tokens_to_ids(d2["tokens"][i2]), device=H.DEVICE)
    h = torch.tensor(d3["hid"]["orig"][i3].astype(np.float32), device=H.DEVICE)[:-1]
    lg = h @ E.T; p = torch.softmax(lg, -1)
    Ep = p @ E; pl = (p * lg).sum(-1, keepdim=True)
    m = (p * lg) @ E - Ep * pl  # Cov_p(E w, logits) = w . m
    var = ((p * lg * lg).sum(-1, keepdim=True) - pl ** 2).clamp_min(1e-3)
    ut = lg.gather(-1, ids[1:, None]) - pl  # centred logit of the target
    xh = h / gF; xh = xh / xh.norm(dim=-1, keepdim=True)
    def perp(v): v = gF * v; return v - xh * (v * xh).sum(-1, keepdim=True)
    q = perp(E[ids[1:]] - Ep); mq = perp(m)
    for l in LAYERS:
        a = torch.tensor(d2["acts"][l][i2].astype(np.float32), device=H.DEVICE)[:-1]
        y = a @ W[l].T; ny = y.norm(dim=-1, keepdim=True); rms = ny / 640 ** 0.5; yh = y / ny
        def eff(qq):  # first-order effect of REMOVING each neuron on the scalar read out by qq
            qg = qq * gp[l]
            return -(a * (qg @ W[l]) - a * ((qg * yh).sum(-1, keepdim=True) * (yh @ W[l]))) / rms
        g_tot = eff(q)
        g_temp = eff(mq) / var * ut  # part explained by a change of temperature
        s = acc.setdefault((grp, l), [0, 0, 0]); s[0] = s[0] + g_tot.sum(0); s[1] = s[1] + (g_tot - g_temp).sum(0); s[2] += a.shape[0]
pickle.dump({k: (v[0].cpu(), v[1].cpu(), v[2]) for k, v in acc.items()}, open("sv/runs/b10-test/work/out/an9.pkl", "wb"))
G = ["sv", "svdevA", "svdevB", "en", "enself", "da", "no"]
rows = []
for l in LAYERS:
    for n in range(2048):
        r = {g: (float(acc[(g, l)][0][n] / acc[(g, l)][2]), float(acc[(g, l)][1][n] / acc[(g, l)][2])) for g in G}
        rows.append((l, n, r))
pickle.dump(rows, open("sv/runs/b10-test/work/out/an9_rows.pkl", "wb"))
# check against E2: sum over its neurons of total vs temperature-free gain
e2 = set(pickle.load(open("sv/runs/b10-test/work/out/sel_e2.pkl", "rb")))
for g in G:
    print(f"E2 neurons {g:7s} total {sum(r[g][0] for l, n, r in rows if (l, n) in e2):+8.1f}  temp-free {sum(r[g][1] for l, n, r in rows if (l, n) in e2):+8.1f}")
rows.sort(key=lambda x: -(x[2]["sv"][1] + x[2]["svdevA"][1]) / 2)
for l, n, r in rows[:20]:
    print(f"L{l} n{n:4d} " + " ".join(f"{g} {r[g][1]:+6.1f}" for g in G))
