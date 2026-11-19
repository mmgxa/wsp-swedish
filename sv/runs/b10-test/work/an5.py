"""Original model on Swedish text (pass-3 orig hidden): where is the loss, how much mass goes to Danish/Norwegian or
English counterparts of the correct Swedish word?"""
import sys, pickle, torch, numpy as np, math; sys.path.insert(0, "sv"); sys.path.insert(0, "sv/runs/b10-test/work")
import harness as H, probe_lib as P, lens as L
d3 = pickle.load(open("sv/runs/b10-test/work/out/p3_E1.pkl", "rb")); its = d3["items"]; tok = H.tokenizer()
cd = {}; ce = {}
for s, o in L.SV_DANO: cd.setdefault(L.t1(s), []).append(L.t1(o))
for s, o in L.SV_EN: ce.setdefault(L.t1(s), []).append(L.t1(o))
st = {"n": 0, "pd": [], "pe": [], "pt": [], "nll": [], "first_nll": [], "nll_rest": []}
for i, it in enumerate(its):
    if it["group"] not in ("lm_devA", "lm_devB", "par_sv"): continue
    ids = tok.convert_tokens_to_ids(d3["tokens"][i])
    lp = P.logp(d3["hid"]["orig"][i])[:-1]
    tg = torch.tensor(ids[1:], device=H.DEVICE)
    nll = -lp.gather(-1, tg[:, None])[:, 0]
    st["first_nll"] += nll[:2].tolist(); st["nll_rest"] += nll[2:].tolist()
    for t, x in enumerate(ids[1:]):
        if x in cd:
            st["pt"].append(float(lp[t, x].exp())); st["pd"].append(float(sum(lp[t, y].exp() for y in cd[x])))
            st["pe"].append(float(sum(lp[t, y].exp() for y in ce.get(x, []))))
print(f"Swedish function-word positions: {len(st['pt'])}; mean p(correct sv) {np.mean(st['pt']):.3f}, p(da/no form) {np.mean(st['pd']):.4f}, p(en form) {np.mean(st['pe']):.4f}")
print(f"nll first 2 tokens {np.mean(st['first_nll']):.2f}, rest {np.mean(st['nll_rest']):.2f} nats/token")
