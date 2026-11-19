"""Free weight analysis (no forward pass).

1. A Swedish-vs-English output direction from the (tied) embedding matrix: mean embedding of Swedish function words
   minus mean embedding of English function words (single-token words, with and without a leading space).
2. For every layer: how strongly attention (o_proj) and MLP (down_proj) can write along that direction, after the
   post-norm gain (1 + w). Also the norm gains themselves. Prints compact tables only.
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "sv"))
import harness as H  # noqa: E402

torch.manual_seed(0)
tok = H.tokenizer()
E = H.orig_tensor("model.embed_tokens.weight")  # [V, 640], read only
Ec = E - E.mean(0)

SV = "och att är inte jag ett av från också någon mycket när där hade blev sig vi det som för på med den har var till eller kan om så men de man efter under över sin sina vid ut upp mer än hur vad vem vilken vilka".split()
EN = "the and is of to in that it was for with as on are this be by from or an at not have had which but they his her their been has were will would can what who how when where".split()


def ids(words):
    out = []
    for w in words:
        for s in (w, " " + w, w.capitalize(), " " + w.capitalize()):
            t = tok(s, add_special_tokens=False).input_ids
            if len(t) == 1:
                out.append(t[0])
    return sorted(set(out))


sv_ids, en_ids = ids(SV), ids(EN)
shared = set(sv_ids) & set(en_ids)
sv_ids = [i for i in sv_ids if i not in shared]
en_ids = [i for i in en_ids if i not in shared]
d = Ec[sv_ids].mean(0) - Ec[en_ids].mean(0)
d = d / d.norm()
print(f"sv tokens {len(sv_ids)}, en tokens {len(en_ids)}")
proj = Ec @ d
print(f"projection on d: sv mean {proj[sv_ids].mean():+.3f}, en mean {proj[en_ids].mean():+.3f}, all-vocab sd {proj.std():.3f}")
torch.save({"d": d, "sv_ids": sv_ids, "en_ids": en_ids}, Path(__file__).with_name("lang_dir.pt"))

print("layer  attn|d^T Wo|/|Wo|  mlp|d^T Wd|/|Wd|  mean(1+post_attn)  mean(1+post_ff)  max-neuron d-align (idx)")
for l in range(18):
    p = f"model.layers.{l}."
    Wo = H.orig_tensor(p + "self_attn.o_proj.weight")      # [640, 1024]
    Wd = H.orig_tensor(p + "mlp.down_proj.weight")         # [640, 2048]
    ga = 1 + H.orig_tensor(p + "post_attention_layernorm.weight")
    gf = 1 + H.orig_tensor(p + "post_feedforward_layernorm.weight")
    # output after the post-norm gain, direction of each column; random baseline is 1/sqrt(640)=0.040
    Wo_g, Wd_g = ga[:, None] * Wo, gf[:, None] * Wd
    ra = (d @ Wo_g).norm() / Wo_g.norm()
    rm = (d @ Wd_g).norm() / Wd_g.norm()
    cos = (d @ Wd_g) / Wd_g.norm(dim=0)
    j = int(cos.abs().argmax())
    print(f"{l:5d}  {ra:16.3f}  {rm:16.3f}  {ga.mean():17.2f}  {gf.mean():15.2f}  {cos[j]:+.3f} ({j})")
