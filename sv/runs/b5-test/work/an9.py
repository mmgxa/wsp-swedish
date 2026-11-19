import sys, json, pickle, numpy as np
sys.path.insert(0, "sv")
import harness as H
meta = json.load(open(H.run_dir() / "work/pass2_meta.json"))
r = pickle.load(open([p for p in (H.run_dir() / "cache").glob("*.pkl") if p.stat().st_size > 1e8][0], "rb"))
A = r["orig"]["mlp"]; T = r["orig"]["tokens"]
for k, m in enumerate(meta):
    if m["i"] in ("gen-15", "gen-24", "gen-29", "own-21", "own-16"):
        n = m["n_prompt"]
        print(m["i"], " ".join(f"{T[k][p]}[{A[15][k][p,459]:+.1f}/{A[14][k][p,1594]:+.1f}]" for p in range(n - 2, n + 9)))
