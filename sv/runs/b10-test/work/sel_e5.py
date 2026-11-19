"""Select E5 (rules fixed before pass 7), after E4 showed that neurons whose removal sharpens the distribution hurt:
layers 12-17, mean |act| Swedish >= 2x English, BOTH total and temperature-free first-order removal gains (an9.py) > 0
on each of sv parallel, DEV lm A, DEV lm B with temperature-free sum > 10, English total and temperature-free >= -1. Scale 0."""
import sys, pickle, numpy as np; sys.path.insert(0, "sv")
rows = pickle.load(open("sv/runs/b10-test/work/out/an9_rows.pkl", "rb"))
d = pickle.load(open("sv/runs/b10-test/work/out/p2.pkl", "rb")); its = d["items"]
def mabs(l, langs):
    return np.concatenate([np.abs(d["acts"][l][i][1:].astype(np.float32)) for i, x in enumerate(its) if x["lang"] in langs]).mean(0)
M = {l: (mabs(l, ["sv", "svdev"]), mabs(l, ["en", "enself"])) for l in range(12, 18)}
SW, EN = ["sv", "svdevA", "svdevB"], ["en", "enself"]
sel = []
for l, n, r in rows:
    if (min(r[g][k] for g in SW for k in (0, 1)) > 0 and sum(r[g][1] for g in SW) > 10
            and min(r[g][k] for g in EN for k in (0, 1)) >= -1 and M[l][0][n] >= 2 * M[l][1][n]):
        sel.append((l, n, r))
sel.sort(key=lambda x: -sum(x[2][g][1] for g in SW))
for l, n, r in sel:
    print(f"L{l} n{n:4d} " + " ".join(f"{g} {r[g][0]:+6.1f}/{r[g][1]:+6.1f}" for g in SW + EN) + f" | |act| sv {M[l][0][n]:.3f} en {M[l][1][n]:.3f}")
pickle.dump([(l, n) for l, n, _ in sel], open("sv/runs/b10-test/work/out/sel_e5.pkl", "wb"))
print(sorted((l, n) for l, n, _ in sel))
