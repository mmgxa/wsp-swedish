"""Rank neurons by first-order gain in log p(target) from removing them (an3.pkl), per text group."""
import sys, pickle, torch; sys.path.insert(0, "sv")
acc = pickle.load(open("sv/runs/b10-test/work/out/an3.pkl", "rb"))
G = ["sv", "svdev", "en", "enself", "da", "no"]
def per_tok(g, l): s, n = acc[(g, l)]; return -s / n  # removal gain per token (unscaled; ~1/200 nats)
rows = []
for l in range(12, 18):
    R = {g: per_tok(g, l) for g in G}
    for n in range(2048):
        rows.append((l, n, {g: float(R[g][n]) for g in G}))
# selection on sv parallel + svdev A-half is not separable here (svdev pooled); use sv parallel for selection, svdev as check
rows.sort(key=lambda x: -x[2]["sv"])
print("top removal gain on Swedish parallel text (unscaled per token):")
for l, n, r in rows[:25]:
    print(f"L{l} n{n:4d} " + " ".join(f"{g} {r[g]:+6.2f}" for g in G))
rows.sort(key=lambda x: -(x[2]["svdev"] + x[2]["sv"]) / 2 + max(0, -(x[2]["en"] + x[2]["enself"]) / 2))
print("top by sv gain minus English loss:")
for l, n, r in rows[:25]:
    print(f"L{l} n{n:4d} " + " ".join(f"{g} {r[g]:+6.2f}" for g in G))
