"""The edit: silence 19 MLP neurons (layers 12-17) whose direct effect lowers the logit of the true next token in
Swedish text, consistently across texts, while having almost no direct effect on English text.

Each neuron i of layer l is silenced by zeroing column i of model.layers.l.mlp.down_proj.weight (its write vector).
The list comes from work/select_neurons.py (pass-1 MLP probe of the original model, unigram-centred direct logit
attribution; selection criterion written there). Deterministic; uses only the original weights.

    uv run --offline python sv/runs/b2-test/work/edit.py
"""
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "sv"))
import harness as H  # noqa: E402

NEURONS = [(17, 798), (17, 666), (16, 999), (13, 1771), (12, 855), (17, 1143), (16, 1573), (17, 458), (16, 598),
           (16, 754), (14, 973), (16, 160), (17, 888), (17, 216), (17, 244), (14, 1494), (17, 170), (16, 1403),
           (17, 1188)]
SCALE = 0.0  # 0 = remove the neuron's output completely

H.reset_edit()
by_layer = defaultdict(list)
for l, n in NEURONS:
    by_layer[l].append(n)
for l in sorted(by_layer):
    name = f"model.layers.{l}.mlp.down_proj.weight"
    W = H.orig_tensor(name)
    W[:, by_layer[l]] *= SCALE
    st = H.save_edit(name, W)
    print(f"{name}: neurons {sorted(by_layer[l])} -> changed {st['changed_after_bf16']} weights")
s = H.edit_summary()
print(f"tensors touched {s['tensors']}, weights changed {s['changed_weights']} ({s['pct_changed']:.4f}% of {s['total_weights']})")
