import sys
sys.path.insert(0, "sv"); sys.path.insert(0, "sv/runs/b5-test/work")
import harness as H
from common import gen_items
items = gen_items()
r = H.fwd("generate", items, which="both", note="pass5: gen both, zero L14n1594,L15n459")
H.show(r)
for it, o, e in zip(items, r["orig"]["per_item"], r["edit"]["per_item"]):
    if o["text"] != e["text"]:
        print(f"{it['id']:7s} [{o['lang']}->{e['lang']}] O: {o['text'][:110]!r}\n{'':7s}           E: {e['text'][:110]!r}")
print("identical answers:", sum(o["text"] == e["text"] for o, e in zip(r["orig"]["per_item"], r["edit"]["per_item"])), "/", len(items))
