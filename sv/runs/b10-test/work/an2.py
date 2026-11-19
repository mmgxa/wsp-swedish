"""Analysis of pass-2 MLP activations (no model calls): direct effect of each neuron on the observed next token."""
import sys, pickle, torch, numpy as np; sys.path.insert(0, "sv"); sys.path.insert(0, "sv/runs/b10-test/work")
import harness as H, lens as L
d = pickle.load(open("sv/runs/b10-test/work/out/p2.pkl", "rb"))
tok = H.tokenizer(); its = d["items"]
ids = [tok.convert_tokens_to_ids(t) for t in d["tokens"]]
E = L.E; Ebar = E.mean(0); gF = L.gF
groups = {"sv": [i for i, x in enumerate(its) if x["lang"] == "sv"], "en": [i for i, x in enumerate(its) if x["lang"] == "en"],
          "da": [i for i, x in enumerate(its) if x["lang"] == "da"], "no": [i for i, x in enumerate(its) if x["lang"] == "no"],
          "devA": [i for i, x in enumerate(its) if x["lang"] == "svdev"][:15], "devB": [i for i, x in enumerate(its) if x["lang"] == "svdev"][15:],
          "enself": [i for i, x in enumerate(its) if x["lang"] == "enself"]}
out = {}
for l in range(18):
    g = 1 + H.orig_tensor(f"model.layers.{l}.post_feedforward_layernorm.weight")
    W = H.orig_tensor(f"model.layers.{l}.mlp.down_proj.weight")
    V = (gF * g)[:, None] * W  # [640, 2048]
    res = {}
    for gname, idx in groups.items():
        C, A = [], []
        for i in idx:
            a = torch.tensor(d["acts"][l][i].astype(np.float32))  # [T, 2048]
            y = a @ W.T; rms = y.pow(2).mean(-1).add(1e-6).sqrt()  # [T]
            tg = torch.tensor(ids[i][1:])
            U = (E[tg] - Ebar) @ V  # [T-1, 2048] logit-diff per unit activation
            C.append(a[:-1] * U / rms[:-1, None]); A.append(a)
        res[gname] = (torch.cat(C).mean(0), torch.cat(A).mean(0), torch.cat(A).abs().mean(0))
    out[l] = res
pickle.dump(out, open("sv/runs/b10-test/work/out/an2.pkl", "wb"))
for l in range(18):
    r = out[l]; selA = r["devA"][0] + r["sv"][0]
    top = torch.topk(-selA, 5).indices
    print(f"L{l:2d} sum DE sv={float(r['sv'][0].sum()):+7.2f} devA={float(r['devA'][0].sum()):+7.2f} en={float(r['enself'][0].sum()):+7.2f}")
    for n in top.tolist():
        print(f"     n{n:4d} DE sv {r['sv'][0][n]:+.3f} devA {r['devA'][0][n]:+.3f} devB {r['devB'][0][n]:+.3f} en {r['en'][0][n]:+.3f} enself {r['enself'][0][n]:+.3f} da {r['da'][0][n]:+.3f} no {r['no'][0][n]:+.3f}")
