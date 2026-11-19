import sys, json, pickle; sys.path.insert(0, "sv"); sys.path.insert(0, "sv/runs/b10-test/work")
import harness as H, items as S
it = H.dev_items("gen", 30) + S.gen_items(S.SV_PROMPTS, "sv_self") + S.gen_items(S.EN_PROMPTS, "en_self")
r = H.fwd("generate", it, which="orig", note="P1 baseline generations: language and repetition")
H.show(r)
pickle.dump({"items": it, "res": r}, open("sv/runs/b10-test/work/out/p1.pkl", "wb"))
for x, y in zip(it, r["orig"]["per_item"]):
    print(f"[{y['lang']}] {x['id']}: {x['prompt'][:60]!r} -> {y['text'][:150]!r}")
