import sys, pickle; sys.path.insert(0, "sv"); sys.path.insert(0, "sv/runs/b10-test/work")
import harness as H, items as S
it = H.dev_items("lm", 30) + H.dev_items("scala", 30) + S.en_lm_items() + \
     [{"id": f"par_sv-{i}", "task": "lm_par_sv", "kind": "lm", "text": p[0]} for i, p in enumerate(S.PARALLEL)]
r = H.fwd("score", it, which="both", note="P9 E3 official scorer both")
H.show(r)
pickle.dump({"items": it, "res": r}, open("sv/runs/b10-test/work/out/p9.pkl", "wb"))
import numpy as np, math
po, pe = r["orig"]["per_item"][:30], r["edit"]["per_item"][:30]
d = np.array([(b["nll"] - a["nll"]) / math.log(2) / a["n_chars"] for a, b in zip(po, pe)])
print(f"DEV lm per-item bpc change: mean {d.mean():+.4f}, items improved {int((d < 0).sum())}/30, worse {int((d > 0).sum())}/30")
