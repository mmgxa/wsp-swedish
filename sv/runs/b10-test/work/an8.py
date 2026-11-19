"""Original model on DEV lm Swedish text: what does it predict when it is wrong? (pass-3 orig hidden)"""
import sys, pickle, torch, numpy as np, re; sys.path.insert(0, "sv"); sys.path.insert(0, "sv/runs/b10-test/work")
import harness as H, probe_lib as P
d = pickle.load(open("sv/runs/b10-test/work/out/p3_E1.pkl", "rb")); its = d["items"]; tok = H.tokenizer()
ex = []; cat = {"same_word_start": 0, "other": 0}; n = 0; top1 = 0; nonascii_mass = []
SV = re.compile(r"[åäöÅÄÖ]")
for i, it in enumerate(its):
    if it["group"] not in ("lm_devA", "lm_devB"): continue
    ids = tok.convert_tokens_to_ids(d["tokens"][i])
    lp = P.logp(d["hid"]["orig"][i])[:-1]
    am = lp.argmax(-1).tolist()
    for t, (y, p) in enumerate(zip(ids[1:], am)):
        n += 1; top1 += int(y == p)
        if y != p and len(ex) < 400: ex.append((tok.convert_ids_to_tokens(y), tok.convert_ids_to_tokens(p)))
print(f"top-1 accuracy on DEV lm: {top1 / n:.3f} over {n} tokens")
rng = np.random.default_rng(0)
for k in rng.choice(len(ex), 40, replace=False):
    print(repr(ex[k][0]), "->", repr(ex[k][1]), end=" | ")
