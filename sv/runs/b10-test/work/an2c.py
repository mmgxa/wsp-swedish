"""Stats of candidate neuron L17 n1977 (pass-2 data): activation spread, share of the MLP output, rms change if removed."""
import sys, pickle, torch, numpy as np; sys.path.insert(0, "sv")
import harness as H
d = pickle.load(open("sv/runs/b10-test/work/out/p2.pkl", "rb")); its = d["items"]
for l, n in [(17, 1977), (16, 52), (17, 1026), (17, 798)]:
    W = H.orig_tensor(f"model.layers.{l}.mlp.down_proj.weight")
    for lang in ["sv", "en", "svdev", "enself"]:
        A = torch.cat([torch.tensor(d["acts"][l][i].astype(np.float32))[1:] for i, x in enumerate(its) if x["lang"] == lang])
        y = A @ W.T; yr = y - A[:, n:n+1] * W[:, n][None]
        a = A[:, n]
        print(f"L{l} n{n} {lang:6s} act mean {a.mean():+.2f} std {a.std():.2f} | |W_n a|/|y| {float((a.abs()*W[:, n].norm()).mean()/y.norm(dim=1).mean()):.3f} | rms(y without n)/rms(y) {float((yr.norm(dim=1)/y.norm(dim=1)).mean()):.3f}")
