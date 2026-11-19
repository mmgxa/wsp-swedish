"""The edit: silence two Norwegian/Danish-language MLP neurons by zeroing their down_proj output columns.

L15 neuron 459 fires on Norwegian text (mean act +1.07) and L14 neuron 1594 on Danish text (+0.70); both stay near 0
(|act| <= 0.11) on Swedish and English text. Read through the unembedding, both output columns favour
Norwegian/Danish-only tokens over Swedish-only tokens. In the original model's Norwegian/Danish drift answers to
Swedish prompts, L15 n459 fires at the token where the answer turns Norwegian ("Det" -> " er", "\\n" -> "Jeg").
Zeroing a column removes that neuron's write to the residual stream; elsewhere it changes nothing.
Deterministic, built from the original weights only."""
import sys
sys.path.insert(0, "sv")
import harness as H

S = 0.0                                   # scale of the output columns (0 = neuron removed)
NEURONS = [(14, 1594), (15, 459)]         # (layer, neuron index in the 2048-wide MLP)

H.reset_edit()
for layer, idx in NEURONS:
    name = f"model.layers.{layer}.mlp.down_proj.weight"
    W = H.orig_tensor(name)
    W[:, idx] *= S
    st = H.save_edit(name, W)
    print(name, "column", idx, "x", S, "-> weights changed:", st["changed_after_bf16"])
s = H.edit_summary()
print("tensors touched:", s["tensors"], "weights changed:", s["changed_weights"], f"({s['pct_changed']:.5f}%)")
