import sys, json
sys.path.insert(0, "sv"); sys.path.insert(0, "sv/runs/b5-test/work")
import harness as H
from texts import PAR
items = H.dev_items("lm", 30) + H.dev_items("scala", 30)
for i, p in enumerate(PAR):
    for l in ("sv", "no", "da", "en"):
        items.append({"id": f"par{i}-{l}", "task": f"par_{l}", "kind": "lm", "text": p[l]})
note = sys.argv[1] if len(sys.argv) > 1 else ""
r = H.fwd("score", items, which="both", note=note)
H.show(r)
import numpy as np
for t in ("lm", "par_sv", "par_en"):
    d = [(e["nll"] - o["nll"]) for it, o, e in zip(items, r["orig"]["per_item"], r["edit"]["per_item"]) if it["task"] == t]
    print(t, "items with lower nll after edit:", sum(x < 0 for x in d), "/", len(d))
sc = [(e["choice_nll"] - o["choice_nll"], o["correct"], e["correct"]) for it, o, e in zip(items, r["orig"]["per_item"], r["edit"]["per_item"]) if it["task"] == "scala"]
print("scala flips wrong->right", sum((not a) and b for _, a, b in sc), "right->wrong", sum(a and (not b) for _, a, b in sc))
