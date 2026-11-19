import sys, pickle
sys.path.insert(0, "sv"); sys.path.insert(0, "sv/runs/b5-test/work")
import harness as H
from common import gen_items
items = gen_items()
r = H.fwd("generate", items, which="orig", note="pass1: diagnose orig generations")
H.show(r)
for it, p in zip(items, r["orig"]["per_item"]):
    print(f"[{p['lang']:5s}] {it['prompt'][:60]!r} -> {p['text'][:150]!r}")
