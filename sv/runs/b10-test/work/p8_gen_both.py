import sys, pickle; sys.path.insert(0, "sv"); sys.path.insert(0, "sv/runs/b10-test/work")
import harness as H, items as S
it = H.dev_items("gen", 30) + S.gen_items(S.SV_PROMPTS, "sv_self") + S.gen_items(S.EN_PROMPTS, "en_self")
r = H.fwd("generate", it, which="both", note="P8 E3 generations both")
H.show(r)
pickle.dump({"items": it, "res": r}, open("sv/runs/b10-test/work/out/p8.pkl", "wb"))
nd = 0
for x, a, b in zip(it, r["orig"]["per_item"], r["edit"]["per_item"]):
    if a["text"] != b["text"]:
        nd += 1
        print(f"{x['id']}: [{a['lang']}] {a['text'][:110]!r}\n      -> [{b['lang']}] {b['text'][:110]!r}")
print(nd, "of", len(it), "answers differ")
