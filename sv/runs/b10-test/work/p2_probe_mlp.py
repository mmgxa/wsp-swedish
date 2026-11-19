import sys, pickle, numpy as np; sys.path.insert(0, "sv"); sys.path.insert(0, "sv/runs/b10-test/work")
import harness as H, items as S
its = []
for li, lang in enumerate(["sv", "en", "da", "no"]):
    its += [{"text": p[li], "lang": lang} for p in S.PARALLEL]
its += [{"text": x["text"], "lang": "svdev"} for x in H.dev_items("lm", 30)]
its += [{"text": t, "lang": "enself"} for t in S.EN_TEXTS]
req = [{"text": x["text"]} for x in its]
r = H.fwd("probe", req, which="orig", what="mlp", layers=list(range(18)), positions="all", note="P2 mlp acts sv/en/da/no + DEV lm")
acts = {l: [a.astype(np.float16) for a in r["orig"]["mlp"][l]] for l in range(18)}
pickle.dump({"items": its, "tokens": r["orig"]["tokens"], "acts": acts}, open("sv/runs/b10-test/work/out/p2.pkl", "wb"))
print(len(its), sum(len(t) for t in r["orig"]["tokens"]), "tokens")
