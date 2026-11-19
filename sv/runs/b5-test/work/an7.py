import sys, json, pickle, numpy as np
sys.path.insert(0, "sv")
import harness as H
meta = json.load(open(H.run_dir() / "work/pass2_meta.json"))
r = [p for p in (H.run_dir() / "cache").glob("*.pkl") if p.stat().st_size > 1e8][0]
A = pickle.load(open(r, "rb"))["orig"]["mlp"]
for l, i in [(14, 1542), (13, 72)]:
    W = H.orig_tensor(f"model.layers.{l}.mlp.down_proj.weight").numpy()
    for g in ("par_sv", "lm_sv", "par_en", "gen_sv"):
        ks = [k for k, m in enumerate(meta) if m["grp"] == g]
        X = np.concatenate([A[l][k][meta[k].get("n_prompt", 1):] for k in ks])
        out = X @ W.T; mine = X[:, i:i+1] * W[:, i][None]
        share = np.linalg.norm(mine, axis=1) / np.linalg.norm(out, axis=1)
        cos = (out * W[:, i]).sum(1) / np.linalg.norm(out, axis=1) / np.linalg.norm(W[:, i])
        print(f"L{l} n{i} {g:7s} share of MLP-out norm: median {np.median(share):.3f} p90 {np.percentile(share,90):.3f}; cos(out, col) median {np.median(cos):+.3f}")
