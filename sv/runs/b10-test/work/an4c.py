"""Activation by language and output-token reading of candidate neurons (pass-2 data + weights)."""
import sys, pickle, torch, re, numpy as np; sys.path.insert(0, "sv"); sys.path.insert(0, "sv/runs/b10-test/work")
import harness as H, lens as L
d = pickle.load(open("sv/runs/b10-test/work/out/p2.pkl", "rb")); its = d["items"]; tok = H.tokenizer()
C = [(17, 1932), (17, 110), (17, 1733), (15, 1226), (17, 1364), (17, 610), (14, 1273), (12, 1529), (16, 1029), (16, 1928), (17, 1067), (17, 435)]
for l, n in C:
    s = []
    for lang in ["sv", "svdev", "da", "no", "en", "enself"]:
        a = np.concatenate([d["acts"][l][i][1:, n].astype(np.float32) for i, x in enumerate(its) if x["lang"] == lang])
        s.append(f"{lang} {a.mean():+.2f}±{a.std():.2f}")
    g = 1 + H.orig_tensor(f"model.layers.{l}.post_feedforward_layernorm.weight")
    v = L.gF * g * H.orig_tensor(f"model.layers.{l}.mlp.down_proj.weight")[:, n]
    sc = L.E @ v
    w = []
    for sign in (1, -1):
        idx = torch.topk(sign * sc, 300).indices.tolist()
        t = [tok.convert_ids_to_tokens(i) for i in idx]
        w.append(" ".join([x for x in t if re.fullmatch(r"▁?[A-Za-zåäöÅÄÖæøÆØ]{2,12}", x)][:10]))
    print(f"L{l} n{n}: " + " | ".join(s) + f" | lens sv-en {float(v @ L.D_en.mean(0)):+.1f} sv-dano {float(v @ L.D_dano.mean(0)):+.1f}")
    print(f"     + {w[0]}\n     - {w[1]}")
