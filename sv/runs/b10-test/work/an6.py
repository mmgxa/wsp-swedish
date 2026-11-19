"""Calibration check on the original model (pass-3 orig hidden): NLL per group if logits were scaled by beta.
Diagnostic only (is the model over/under-confident on Swedish vs English?)."""
import sys, pickle, torch, numpy as np; sys.path.insert(0, "sv"); sys.path.insert(0, "sv/runs/b10-test/work")
import harness as H, probe_lib as P
d3 = pickle.load(open("sv/runs/b10-test/work/out/p3_E1.pkl", "rb")); its = d3["items"]; tok = H.tokenizer()
E = H.orig_tensor("model.embed_tokens.weight").to(H.DEVICE)
B = [0.8, 0.9, 1.0, 1.1, 1.2]
res = {}
for i, it in enumerate(its):
    if it["kind"] != "lm": continue
    ids = torch.tensor(tok.convert_tokens_to_ids(d3["tokens"][i]), device=H.DEVICE)
    lg = torch.tensor(d3["hid"]["orig"][i].astype(np.float32), device=H.DEVICE)[:-1] @ E.T
    r = res.setdefault(it["group"], [np.zeros(len(B)), 0, 0.0])
    for k, b in enumerate(B):
        r[0][k] += float(-torch.log_softmax(lg * b, -1).gather(-1, ids[1:, None]).sum())
    p = torch.softmax(lg, -1); r[2] += float(-(p * torch.log(p + 1e-30)).sum(-1).sum())
    r[1] += len(ids) - 1
for g, (v, n, ent) in res.items():
    print(f"{g:8s} nll/tok at beta {B}: " + " ".join(f"{x / n:.3f}" for x in v) + f" | entropy {ent / n:.2f}")
