"""Rank neurons that hurt the observed next token on Swedish text but not on English (direct effect, pass-2 data)."""
import sys, pickle, torch, re; sys.path.insert(0, "sv"); sys.path.insert(0, "sv/runs/b10-test/work")
import harness as H, lens as L
out = pickle.load(open("sv/runs/b10-test/work/out/an2.pkl", "rb"))
tok = H.tokenizer()
rows = []
for l in range(18):
    r = out[l]
    ssv = (r["sv"][0] + r["devA"][0]) / 2; sen = (r["en"][0] + r["enself"][0]) / 2
    tot = float((r["sv"][0] + r["devA"][0]).sum() / 2)
    for n in range(2048):
        rows.append((float(ssv[n]), float(sen[n]), float(r["devB"][0][n]), float((r["da"][0][n] + r["no"][0][n]) / 2), l, n,
                     float(r["sv"][1][n]), float(r["en"][1][n]), float(r["da"][1][n]), float(r["no"][1][n])))
sel = [x for x in rows if x[1] > 0.2 * x[0]]  # much less harmful (or helpful) on English
sel.sort()
def toks(l, n, k=8):
    g = 1 + H.orig_tensor(f"model.layers.{l}.post_feedforward_layernorm.weight")
    v = L.gF * g * H.orig_tensor(f"model.layers.{l}.mlp.down_proj.weight")[:, n]
    s = L.E @ v
    out = []
    for sign in (1, -1):
        idx = torch.topk(sign * s, 400).indices.tolist()
        w = [tok.convert_ids_to_tokens(i) for i in idx]
        w = [t for t in w if re.fullmatch(r"▁?[A-Za-zåäöÅÄÖæøÆØ]{2,12}", t)][:k]
        out.append(" ".join(w))
    return out, float(v @ (L.D_dano.mean(0))), float(v @ L.D_en.mean(0))
for x in sel[:25]:
    (pos, neg), sd, se = toks(x[4], x[5])
    print(f"L{x[4]:2d} n{x[5]:4d} DE sv {x[0]:+8.1f} en {x[1]:+8.1f} devB {x[2]:+8.1f} dano {x[3]:+8.1f} | act sv {x[6]:+.2f} en {x[7]:+.2f} da {x[8]:+.2f} no {x[9]:+.2f} | lens sv-dano {sd:+.2f} sv-en {se:+.2f}")
    print(f"      + {pos[:90]}\n      - {neg[:90]}")
