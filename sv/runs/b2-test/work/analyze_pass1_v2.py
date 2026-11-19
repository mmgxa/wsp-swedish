"""Free analysis of pass-1 statistics, version 2: DLA centred on the unigram distribution of the texts.

Version 1 centred the true-token embedding on the vocabulary mean. That rewards and punishes neurons that move all
frequent tokens together (a calibration / confidence effect), which lowers the true-token logit but also the softmax
normaliser. Centring on the unigram mean of the texts' own tokens, E[y] - sum_v q_v E[v], removes most of that effect
(first-order change of log p(y) if the model's prediction were the unigram distribution).
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "sv"))
import harness as H  # noqa: E402

W = Path(__file__).parent
z = np.load(W / "pass1_stats.npz")
D, B, A, NT, sv, rid = z["D"], z["B"], z["A"], z["NT"], z["is_sv"], z["ids"]
tok = H.tokenizer()
tiny = {r["id"] for r in H.dev_items("lm")}
rows = [r for r in H.load_jsonl(ROOT / "sv/data/default/dev/lm.jsonl") if r["id"] not in tiny] + \
    H.load_jsonl(ROOT / "sv/data/default/dev/lm_en.jsonl", 50)
assert [r["id"] for r in rows] == list(rid)

E = H.orig_tensor("model.embed_tokens.weight")
Ebar = E.mean(0)
gN = 1 + H.orig_tensor("model.norm.weight")
cent = {}
for lang, mask in (("sv", sv), ("en", ~sv)):
    cnt = Counter()
    for r, m in zip(rows, mask):
        if m:
            cnt.update(([tok.bos_token_id] + H._ids(r["text"]))[2:])
    ids = torch.tensor(list(cnt)); w = torch.tensor([cnt[i] for i in cnt.keys()], dtype=torch.float32)
    cent[lang] = (w[:, None] * E[ids]).sum(0) / w.sum() - Ebar
del E
Dn = D.copy()
for l in range(18):
    p = f"model.layers.{l}."
    M = ((1 + H.orig_tensor(p + "post_feedforward_layernorm.weight")) * gN)[:, None] * H.orig_tensor(p + "mlp.down_proj.weight")
    for lang, mask in (("sv", sv), ("en", ~sv)):
        wl = (cent[lang] @ M).numpy()
        Dn[mask, l] -= B[mask, l] * wl
np.save(W / "pass1_dla_unigram.npy", Dn)

svi, eni = np.where(sv)[0], np.where(~sv)[0]
h1, h2 = svi[0::2], svi[1::2]
pt = lambda idx: Dn[idx].sum(0) / NT[idx].sum()
S, S1, S2, EN = pt(svi), pt(h1), pt(h2), pt(eni)
fneg = (Dn[svi] < 0).mean(0)
actS, actE = A[svi].sum(0) / NT[svi].sum(), A[eni].sum(0) / NT[eni].sum()
tot = Dn[svi].sum((1, 2)) / NT[svi]
print(f"total MLP DLA per Swedish token (unigram-centred) {tot.mean():+.1f}; English {(Dn[eni].sum((1,2))/NT[eni]).mean():+.1f}")
print(f"corr(half1, half2) {np.corrcoef(S1.ravel(), S2.ravel())[0,1]:.3f}; corr(sv, en) {np.corrcoef(S.ravel(), EN.ravel())[0,1]:.3f}")
print("\nmost harmful on Swedish (per token; halves; English; frac sv texts negative; mean act sv/en)")
for f in np.argsort(S, axis=None)[:30]:
    l, n = np.unravel_index(f, S.shape)
    print(f"  L{l:2d} n{n:4d}  sv {S[l,n]:+8.2f}  h1 {S1[l,n]:+8.2f} h2 {S2[l,n]:+8.2f}  en {EN[l,n]:+8.2f}  "
          f"neg {fneg[l,n]:.2f}  act {actS[l,n]:+.3f}/{actE[l,n]:+.3f}")
# Swedish-specific harm: harmful on Swedish, replicates in both halves, not harmful-or-helpful much on English
spec = S - np.maximum(EN, S)  # placeholder to keep shapes; real criterion below
crit = (S1 < 0) & (S2 < 0) & (fneg >= 0.8) & (EN > 0.25 * S)  # English harm at most a quarter of Swedish harm
gap = np.where(crit, S - np.minimum(EN, 0), 0)
print("\nSwedish-specific harmful neurons (sv harm replicates in both halves, English harm < 25% of Swedish)")
for f in np.argsort(gap, axis=None)[:30]:
    l, n = np.unravel_index(f, S.shape)
    print(f"  L{l:2d} n{n:4d}  sv {S[l,n]:+8.2f}  h1 {S1[l,n]:+8.2f} h2 {S2[l,n]:+8.2f}  en {EN[l,n]:+8.2f}  "
          f"neg {fneg[l,n]:.2f}  act {actS[l,n]:+.3f}/{actE[l,n]:+.3f}")
