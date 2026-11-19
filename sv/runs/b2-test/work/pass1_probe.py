"""Pass 1: MLP probe of the ORIGINAL model on Swedish and English text, then per-neuron direct logit attribution.

Items: 70 Swedish DEV texts (default/dev/lm.jsonl minus the 30 tiny-DEV lm items) + 50 English texts (default/dev/lm_en).
For layer l, position t (predicting token t+1), with a = down_proj input (from fwd), the MLP's contribution to the
residual is  c = (1+g_postff) * (Wd a) / rms(Wd a). Neuron i's share is a_i (1+g_postff) Wd[:, i] / rms(Wd a).
Its direct effect on the logit of the true next token y (up to the final-norm scale, unknown and positive) is
    DLA_i,t = share_i . ((1+g_final) * (E[y] - mean_vocab E)).
Position 0 (BOS) is excluded. Saves per-text sums to work/pass1_stats.npz. Re-running is free (served from cache).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "sv"))
import harness as H  # noqa: E402

torch.manual_seed(0)
tiny = {r["id"] for r in H.dev_items("lm")}
sv_rows = [r for r in H.load_jsonl(ROOT / "sv/data/default/dev/lm.jsonl") if r["id"] not in tiny]
en_rows = H.load_jsonl(ROOT / "sv/data/default/dev/lm_en.jsonl", 50)
rows = sv_rows + en_rows
items = [{"text": r["text"], "task": r["task"]} for r in rows]
LAYERS = list(range(18))

res = H.fwd("probe", items, which="orig", what="mlp", layers=LAYERS, positions="all",
            note="MLP acts, all layers/positions: 70 sv texts (not tiny-DEV) + 50 en texts, for per-neuron DLA")
acts = res["orig"]["mlp"]

tok = H.tokenizer()
E = H.orig_tensor("model.embed_tokens.weight")
Ec = E - E.mean(0)
gN = 1 + H.orig_tensor("model.norm.weight")
del E

n_text = len(rows)
D = np.zeros((n_text, 18, 2048), np.float32)      # sum over positions of DLA
A = np.zeros((n_text, 18, 2048), np.float32)      # sum over positions of activation
B = np.zeros((n_text, 18, 2048), np.float32)      # sum over positions of activation / rms(Wd a), for re-centering
NT = np.zeros(n_text, np.int64)
for l in LAYERS:
    p = f"model.layers.{l}."
    Wd = H.orig_tensor(p + "mlp.down_proj.weight")                 # [640, 2048]
    gf = 1 + H.orig_tensor(p + "post_feedforward_layernorm.weight")
    M = (gf * gN)[:, None] * Wd                                    # [640, 2048]
    for k, r in enumerate(rows):
        ids = [tok.bos_token_id] + H._ids(r["text"])
        a = torch.from_numpy(acts[l][k]).float()                   # [T, 2048]
        assert a.shape[0] == len(ids)
        a = a[1:-1]                                                # positions 1..T-2 predict ids[2..T-1]
        y = torch.tensor(ids[2:])
        rms = ((a @ Wd.T).pow(2).mean(-1) + 1e-6).sqrt()           # [T-2]
        S = Ec[y] @ M                                              # [T-2, 2048]
        D[k, l] = (a * S / rms[:, None]).sum(0).numpy()
        A[k, l] = a.sum(0).numpy()
        B[k, l] = (a / rms[:, None]).sum(0).numpy()
        NT[k] = a.shape[0]
is_sv = np.array([r["task"] == "lm" for r in rows])
np.savez_compressed(Path(__file__).with_name("pass1_stats.npz"), D=D, A=A, B=B, NT=NT, is_sv=is_sv,
                    ids=np.array([r["id"] for r in rows]))
print(f"saved stats: {is_sv.sum()} sv texts ({NT[is_sv].sum()} tokens), {(~is_sv).sum()} en texts ({NT[~is_sv].sum()} tokens)")
