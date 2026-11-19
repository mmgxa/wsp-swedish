"""Temperature part (total minus temperature-free first-order gain) of each E4 neuron; negative = removal sharpens."""
import sys, pickle; sys.path.insert(0, "sv")
rows = {(l, n): r for l, n, r in pickle.load(open("sv/runs/b10-test/work/out/an9_rows.pkl", "rb"))}
for l, n in pickle.load(open("sv/runs/b10-test/work/out/sel_e4.pkl", "rb")):
    r = rows[(l, n)]
    print(f"L{l} n{n:4d} " + " ".join(f"{g}: tot {r[g][0]:+6.1f} temp {r[g][0] - r[g][1]:+6.1f}" for g in ["sv", "svdevA", "svdevB"]))
