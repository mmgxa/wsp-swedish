"""First-order direct effect of removing each late-layer neuron on log p(target), with both RMSNorm couplings:
the neuron's output is orthogonalised against the layer's MLP output y (post_feedforward_layernorm) and its residual
contribution against the final residual direction x/rms(x) = h/gF (final norm). Scale 1/rms(x) is unknown per position:
reported unscaled and calibrated by one global factor from the measured E1 effect (pass 3)."""
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
acc = {}; chk = {}
for i2, x in enumerate(d2["items"]):
    i3 = idx3[x["text"]]
    grp = devgrp.get(x["text"], x["lang"])
    ids = torch.tensor(tok.convert_tokens_to_ids(d2["tokens"][i2]), device=H.DEVICE)
    h = torch.tensor(d3["hid"]["orig"][i3].astype(np.float32), device=H.DEVICE)[:-1]
    lp = P.logp(d3["hid"]["orig"][i3])[:-1]; lpe = P.logp(d3["hid"]["edit"][i3])[:-1]
    dt = E[ids[1:]] - lp.exp() @ E  # [T', 640]: d logp(target) / d logits projected to embedding space
    xh = h / gF; xh = xh / xh.norm(dim=-1, keepdim=True)  # unit direction of the final residual
    q = gF * dt  # d logp / d (x/rms(x))
    q = q - xh * (q * xh).sum(-1, keepdim=True)  # final-norm coupling: only the component orthogonal to x matters
    obs = float((lpe.gather(-1, ids[1:, None]) - lp.gather(-1, ids[1:, None])).sum())
    for l in LAYERS:
        a = torch.tensor(d2["acts"][l][i2].astype(np.float32), device=H.DEVICE)[:-1]
        y = a @ W[l].T; ny = y.norm(dim=-1, keepdim=True); rms = ny / 640 ** 0.5
        yh = y / ny
        # removal of neuron n changes normalised output by -(W_n a_n - yh (yh.W_n a_n)) / rms(y), times gp
        qg = q * gp[l]  # [T', 640]
        t1 = a * (qg @ W[l]) / rms  # [T', 2048]
        t2 = a * ((qg * yh).sum(-1, keepdim=True) * (yh @ W[l])) / rms
        gain = -(t1 - t2)  # first-order gain in logp(target) from removing the neuron (unscaled by 1/rms(x))
        s = acc.setdefault((grp, l), [0, 0]); s[0] = s[0] + gain.sum(0); s[1] += gain.shape[0]
        if l == 17:
            c = chk.setdefault(grp, [0, 0, 0]); c[0] += float(gain[:, 1977].sum()); c[1] += obs; c[2] += gain.shape[0]
for g, (pr, ob, n) in chk.items():
    print(f"E1 check {g:7s} predicted(unscaled) {pr / n:+8.3f}/tok observed {ob / n:+.4f} nats/tok ratio {ob / pr:.5f}")
pickle.dump({k: (v[0].cpu(), v[1]) for k, v in acc.items()}, open("sv/runs/b10-test/work/out/an4.pkl", "wb"))
