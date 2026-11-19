"""Free analysis of pass-1 statistics: rank neurons by harmful direct effect on Swedish next tokens.
Selection uses only the 70 Swedish probe texts (split in two halves for a replication check) and the 50 English texts.
"""
from pathlib import Path

import numpy as np

z = np.load(Path(__file__).with_name("pass1_stats.npz"))
D, A, NT, sv = z["D"], z["A"], z["NT"], z["is_sv"]
svi, eni = np.where(sv)[0], np.where(~sv)[0]
h1, h2 = svi[0::2], svi[1::2]


def per_tok(idx):
    return D[idx].sum(0) / NT[idx].sum()


S, S1, S2, EN = per_tok(svi), per_tok(h1), per_tok(h2), per_tok(eni)
fneg = (D[svi] < 0).mean(0)
actS = A[svi].sum(0) / NT[svi].sum()
actE = A[eni].sum(0) / NT[eni].sum()

tot = D[svi].sum((1, 2)) / NT[svi]
print(f"sum of all MLP-neuron DLA on the true Swedish token, per token: mean {tot.mean():+.2f} (sd over texts {tot.std():.2f})")
print("per-layer total DLA per token  sv / en:")
print("  " + "  ".join(f"L{l}:{S[l].sum():+.2f}/{EN[l].sum():+.2f}" for l in range(18)))
print(f"neuron DLA per token: sd {S.std():.4f}; most negative {S.min():+.4f}; most positive {S.max():+.4f}")

order = np.argsort(S, axis=None)
print("\nmost harmful on Swedish (DLA per Swedish token; halves; English; frac of sv texts negative; mean act sv/en)")
for f in order[:40]:
    l, n = np.unravel_index(f, S.shape)
    print(f"  L{l:2d} n{n:4d}  sv {S[l, n]:+.4f}  h1 {S1[l, n]:+.4f} h2 {S2[l, n]:+.4f}  en {EN[l, n]:+.4f}  "
          f"neg {fneg[l, n]:.2f}  act {actS[l, n]:+.3f}/{actE[l, n]:+.3f}")

# replication: of the top-k by half 1, how many are also harmful in half 2?
for k in (20, 50, 100, 200):
    top = np.argsort(S1, axis=None)[:k]
    r = S2.ravel()[top]
    print(f"top {k:3d} by half 1: half-2 negative {np.mean(r < 0):.2f}, half-2 mean {r.mean():+.4f} "
          f"(half-1 mean {S1.ravel()[top].mean():+.4f}), English mean {EN.ravel()[top].mean():+.4f}")
c = np.corrcoef(S1.ravel(), S2.ravel())[0, 1]
ce = np.corrcoef(S.ravel(), EN.ravel())[0, 1]
print(f"corr(half1, half2) over all neurons {c:.3f}; corr(sv, en) {ce:.3f}")
