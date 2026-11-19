"""Select the neurons for the edit from work/pass1_dla_unigram.npy (free; uses only the 70 Swedish + 50 English probe texts).

Criterion (fixed before pass 2):
  - unigram-centred DLA per Swedish token <= -9 (harmful on Swedish),
  - harmful in BOTH halves of the Swedish texts (<= -5 each) and negative on >= 80% of the Swedish texts,
  - English DLA per token within 25% of the Swedish harm in absolute value (|en| <= 0.25 |sv|), so removing the
    neuron should change English little in either direction.
"""
from pathlib import Path

import numpy as np

W = Path(__file__).parent
z = np.load(W / "pass1_stats.npz")
Dn, NT, sv = np.load(W / "pass1_dla_unigram.npy"), z["NT"], z["is_sv"]
svi, eni = np.where(sv)[0], np.where(~sv)[0]
pt = lambda idx: Dn[idx].sum(0) / NT[idx].sum()
S, S1, S2, EN = pt(svi), pt(svi[0::2]), pt(svi[1::2]), pt(eni)
fneg = (Dn[svi] < 0).mean(0)
sel = (S <= -9) & (S1 <= -5) & (S2 <= -5) & (fneg >= 0.8) & (np.abs(EN) <= 0.25 * np.abs(S))
L, N = np.where(sel)
o = np.argsort(S[L, N])
L, N = L[o], N[o]
print(f"{len(L)} neurons; summed DLA per token: sv {S[L, N].sum():+.1f}, en {EN[L, N].sum():+.1f}")
print("per layer:", {int(l): int((L == l).sum()) for l in np.unique(L)})
print("NEURONS =", [(int(l), int(n)) for l, n in zip(L, N)])
