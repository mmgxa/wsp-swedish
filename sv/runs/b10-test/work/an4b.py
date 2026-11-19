"""Rank neurons by first-order removal gain (an4.pkl). Selection groups: sv parallel + DEV lm A; check: DEV lm B."""
import sys, pickle; sys.path.insert(0, "sv")
acc = pickle.load(open("sv/runs/b10-test/work/out/an4.pkl", "rb"))
G = ["sv", "svdevA", "svdevB", "en", "enself", "da", "no"]
rows = []
for l in range(12, 18):
    R = {g: acc[(g, l)][0] / acc[(g, l)][1] for g in G}
    for n in range(2048):
        rows.append((l, n, {g: float(R[g][n]) for g in G}))
def sel_score(r): return (r["sv"] + r["svdevA"]) / 2
def en_score(r): return (r["en"] + r["enself"]) / 2
fmt = lambda l, n, r: f"L{l} n{n:4d} " + " ".join(f"{g} {r[g]:+7.2f}" for g in G)
rows.sort(key=lambda x: -sel_score(x[2]))
print("top by Swedish selection gain:")
for l, n, r in rows[:15]: print(fmt(l, n, r))
print("Swedish-specific (sel > 3, English gain >= -0.25*sel), sorted by sel - max(0,-en):")
sp = [x for x in rows if sel_score(x[2]) > 3 and en_score(x[2]) >= -0.25 * sel_score(x[2])]
sp.sort(key=lambda x: -(sel_score(x[2]) - max(0, -en_score(x[2]))))
for l, n, r in sp[:30]: print(fmt(l, n, r))
pickle.dump(rows, open("sv/runs/b10-test/work/out/an4_rows.pkl", "wb"))
