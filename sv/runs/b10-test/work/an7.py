"""Entropy, NLL and best logit scale per group for orig and edit sides of a probe pass (diagnostic only)."""
import sys, pickle, torch, numpy as np; sys.path.insert(0, "sv"); sys.path.insert(0, "sv/runs/b10-test/work")
import harness as H
tag = sys.argv[1]
d = pickle.load(open(f"sv/runs/b10-test/work/out/{tag}.pkl", "rb")); its = d["items"]; tok = H.tokenizer()
E = H.orig_tensor("model.embed_tokens.weight").to(H.DEVICE)
B = np.round(np.arange(0.4, 1.21, 0.05), 2)
res = {}
for i, it in enumerate(its):
    if it["kind"] != "lm": continue
    ids = torch.tensor(tok.convert_tokens_to_ids(d["tokens"][i]), device=H.DEVICE)
    for s in ("orig", "edit"):
        lg = torch.tensor(d["hid"][s][i].astype(np.float32), device=H.DEVICE)[:-1] @ E.T
        r = res.setdefault((it["group"], s), [np.zeros(len(B)), 0, 0.0])
        for k, b in enumerate(B):
            r[0][k] += float(-torch.log_softmax(lg * b, -1).gather(-1, ids[1:, None]).sum())
        p = torch.softmax(lg, -1); r[2] += float(-(p * torch.log(p + 1e-30)).sum(-1).sum()); r[1] += len(ids) - 1
for (g, s), (v, n, ent) in res.items():
    k = int(np.argmin(v)); one = int(np.where(B == 1.0)[0][0])
    print(f"{g:8s} {s:4s} nll {v[one] / n:.3f} entropy {ent / n:.2f} best beta {B[k]:.2f} -> nll {v[k] / n:.3f}")
