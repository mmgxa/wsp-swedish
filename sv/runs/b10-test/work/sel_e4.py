"""Select E4 (rules fixed before pass 6), a relaxed version of E3 after E3 was confirmed in pass 5:
layers 12-17, mean |act| Swedish >= 2x English, temperature-free first-order removal gain (an9.py) > 0 on each of
sv parallel, DEV lm A, DEV lm B with sum > 10, English temperature-free gain >= -1 on both English groups. Scale 0."""
import sys, pickle, numpy as np; sys.path.insert(0, "sv")
rows = pickle.load(open("sv/runs/b10-test/work/out/an9_rows.pkl", "rb"))
d = pickle.load(open("sv/runs/b10-test/work/out/p2.pkl", "rb")); its = d["items"]
def mabs(l, langs):
    return np.concatenate([np.abs(d["acts"][l][i][1:].astype(np.float32)) for i, x in enumerate(its) if x["lang"] in langs]).mean(0)
M = {l: (mabs(l, ["sv", "svdev"]), mabs(l, ["en", "enself"])) for l in range(12, 18)}
sel = []
for l, n, r in rows:
    tf = {g: v[1] for g, v in r.items()}; sw = [tf["sv"], tf["svdevA"], tf["svdevB"]]
    if min(sw) > 0 and sum(sw) > 10 and min(tf["en"], tf["enself"]) >= -1 and M[l][0][n] >= 2 * M[l][1][n]:
        sel.append((l, n, tf))
sel.sort(key=lambda x: -(x[2]["sv"] + x[2]["svdevA"] + x[2]["svdevB"]))
for l, n, tf in sel:
    print(f"L{l} n{n:4d} " + " ".join(f"{g} {v:+6.1f}" for g, v in tf.items()) + f" | |act| sv {M[l][0][n]:.3f} en {M[l][1][n]:.3f}")
print(len(sel), "neurons; sums:", {g: round(sum(x[2][g] for x in sel), 1) for g in sel[0][2]})
pickle.dump([(l, n) for l, n, _ in sel], open("sv/runs/b10-test/work/out/sel_e4.pkl", "wb"))
