import sys, json, pickle, numpy as np
sys.path.insert(0, "sv")
import harness as H
meta = json.load(open(H.run_dir() / "work/pass2_meta.json"))
r = [p for p in (H.run_dir() / "cache").glob("*.pkl") if p.stat().st_size > 1e8][0]
r = pickle.load(open(r, "rb"))
A = r["orig"]["mlp"]; toks = r["orig"]["tokens"]
neur = [(14, 1542), (13, 72), (17, 700), (15, 459), (15, 1944)]
rows = {}
for k, m in enumerate(meta):
    if not m["grp"].startswith("gen_"): continue
    np_ = m["n_prompt"]
    # positions: prompt-mid (mean over user content), last 3 prompt tokens, first 3 answer tokens
    v = []
    for l, i in neur:
        a = A[l][k][:, i]
        v.append((a[4:np_-5].mean(), a[np_-1], a[np_:np_+3].mean()))
    rows[k] = v
    lab = m["grp"][4:]
    if lab != "sv" or k % 7 == 0:
        print(f"{m['i']:7s} {lab:5s} " + " | ".join(f"{x:+.2f} {y:+.2f} {z:+.2f}" for x, y, z in v), repr("".join(toks[k][np_:np_+4]))[:40])
print("cols per neuron: mean over user prompt, at last prompt token, first 3 answer tokens; neurons", neur)
for lab in ("sv", "no", "da", "en", "other"):
    ks = [k for k, m in enumerate(meta) if m["grp"] == "gen_" + lab]
    if ks:
        print(lab, len(ks), " | ".join(" ".join(f"{np.mean([rows[k][j][c] for k in ks]):+.2f}" for c in range(3)) for j in range(len(neur))))
