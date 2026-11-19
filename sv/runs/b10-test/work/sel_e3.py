"""Select E3 (rules fixed before pass 5): Scandinavian-only neurons (mean |act| Swedish >= 3x English, layers 12-17)
whose temperature-free first-order removal gain (an9.py) is > 2 on each of: sv parallel, DEV lm A, DEV lm B, and whose
English temperature-free gain is >= -1 on both English groups. Scale 0 (small neurons: removal is still a small change)."""
import sys, pickle, numpy as np; sys.path.insert(0, "sv")
rows = pickle.load(open("sv/runs/b10-test/work/out/an9_rows.pkl", "rb"))
d = pickle.load(open("sv/runs/b10-test/work/out/p2.pkl", "rb")); its = d["items"]
def mabs(l, langs):
    return np.concatenate([np.abs(d["acts"][l][i][1:].astype(np.float32)) for i, x in enumerate(its) if x["lang"] in langs]).mean(0)
M = {l: (mabs(l, ["sv", "svdev"]), mabs(l, ["en", "enself"])) for l in range(12, 18)}
sel = []
for l, n, r in rows:
    tf = {g: v[1] for g, v in r.items()}
    if min(tf["sv"], tf["svdevA"], tf["svdevB"]) > 2 and min(tf["en"], tf["enself"]) >= -1 and M[l][0][n] >= 3 * M[l][1][n]:
        sel.append((l, n, tf))
sel.sort(key=lambda x: -min(x[2]["sv"], x[2]["svdevA"], x[2]["svdevB"]))
for l, n, tf in sel:
    print(f"L{l} n{n:4d} " + " ".join(f"{g} {v:+6.1f}" for g, v in tf.items()) + f" | |act| sv {M[l][0][n]:.3f} en {M[l][1][n]:.3f}")
print(len(sel), "neurons")
pickle.dump([(l, n) for l, n, _ in sel], open("sv/runs/b10-test/work/out/sel_e3.pkl", "wb"))
