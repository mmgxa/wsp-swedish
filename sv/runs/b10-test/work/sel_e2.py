"""Select E2: Scandinavian-conditional neurons (L12-17) with first-order harm on Swedish next tokens.
Rules fixed in advance: sel=(sv parallel + DEV lm A)/2 > 5; English gain >= -0.1*sel; mean|act| on Swedish >= 3x on English.
DEV lm B is not used for selection, only reported."""
import sys, pickle, numpy as np; sys.path.insert(0, "sv")
rows = pickle.load(open("sv/runs/b10-test/work/out/an4_rows.pkl", "rb"))
d = pickle.load(open("sv/runs/b10-test/work/out/p2.pkl", "rb")); its = d["items"]
dev = [x["text"] for x in __import__("harness").dev_items("lm", 15)]
def mabs(l, langs, texts=None):
    a = [np.abs(d["acts"][l][i][1:].astype(np.float32)) for i, x in enumerate(its) if x["lang"] in langs and (texts is None or x["text"] in texts)]
    return np.concatenate(a).mean(0)
M = {l: (np.maximum(mabs(l, ["sv"]), mabs(l, ["svdev"], set(dev))), mabs(l, ["en", "enself"])) for l in range(12, 18)}
sel = []
for l, n, r in rows:
    s = (r["sv"] + r["svdevA"]) / 2; e = (r["en"] + r["enself"]) / 2
    msv, men = M[l][0][n], M[l][1][n]
    if s > 5 and e >= -0.1 * s and msv >= 3 * men:
        sel.append((l, n, s, e, r["svdevB"], (r["da"] + r["no"]) / 2, msv, men))
sel.sort(key=lambda x: -x[2])
for x in sel: print(f"L{x[0]} n{x[1]:4d} sel {x[2]:+6.1f} en {x[3]:+6.1f} devB {x[4]:+6.1f} dano {x[5]:+6.1f} |act| sv {x[6]:.3f} en {x[7]:.3f}")
print(len(sel), "neurons; sum sel", round(sum(x[2] for x in sel), 1), "sum devB", round(sum(x[4] for x in sel), 1), "sum en", round(sum(x[3] for x in sel), 1))
pickle.dump([(x[0], x[1]) for x in sel], open("sv/runs/b10-test/work/out/sel_e2.pkl", "wb"))
