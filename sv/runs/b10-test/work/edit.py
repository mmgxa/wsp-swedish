"""Deterministic edit, built from the original weights only.

Usage: uv run --offline python sv/runs/b10-test/work/edit.py [candidate]   (default: the frozen candidate)
"""
import sys
sys.path.insert(0, "sv")
import harness as H

# candidate name -> list of (layer, neuron, scale) for MLP down_proj columns
CANDIDATES = {
    "E1": [(17, 1977, 0.0)],
    # E2: 45 Scandinavian-only neurons (L12-17) whose direct output lowers log p(next token) on Swedish (sel_e2.py)
    "E2": [(13, 208, 0.0), (14, 1141, 0.0), (14, 1406, 0.0), (14, 1706, 0.0), (15, 10, 0.0), (15, 151, 0.0), (15, 308, 0.0), (15, 367, 0.0), (15, 408, 0.0), (15, 459, 0.0), (15, 575, 0.0), (15, 610, 0.0), (15, 1274, 0.0), (15, 1404, 0.0), (15, 1468, 0.0), (15, 1501, 0.0), (16, 93, 0.0), (16, 127, 0.0), (16, 292, 0.0), (16, 307, 0.0), (16, 482, 0.0), (16, 851, 0.0), (16, 940, 0.0), (16, 973, 0.0), (16, 1028, 0.0), (16, 1152, 0.0), (16, 1401, 0.0), (16, 1454, 0.0), (16, 1518, 0.0), (16, 1847, 0.0), (16, 1928, 0.0), (16, 2011, 0.0), (17, 255, 0.0), (17, 262, 0.0), (17, 528, 0.0), (17, 624, 0.0), (17, 1067, 0.0), (17, 1095, 0.0), (17, 1262, 0.0), (17, 1364, 0.0), (17, 1411, 0.0), (17, 1629, 0.0), (17, 1697, 0.0), (17, 1849, 0.0), (17, 1903, 0.0)],
    # E3: 3 Scandinavian-only neurons with temperature-free first-order gain on all Swedish groups (sel_e3.py)
    "E3": [(15, 459, 0.0), (17, 736, 0.0), (17, 1067, 0.0)],
    # E4: relaxed E3 rule, 14 neurons (sel_e4.py)
    "E4": [(15, 61, 0.0), (15, 459, 0.0), (15, 575, 0.0), (16, 754, 0.0), (17, 31, 0.0), (17, 52, 0.0), (17, 242, 0.0),
           (17, 262, 0.0), (17, 624, 0.0), (17, 736, 0.0), (17, 925, 0.0), (17, 1067, 0.0), (17, 1243, 0.0), (17, 1364, 0.0)],
    # E5: E3 + L17 n1364 (sel_e5.py: total and temperature-free gains both > 0 on all Swedish groups)
    "E5": [(15, 459, 0.0), (17, 736, 0.0), (17, 1067, 0.0), (17, 1364, 0.0)],
}
FROZEN = "E3"


def build(name):
    H.reset_edit()
    by_layer = {}
    for l, n, s in CANDIDATES[name]:
        by_layer.setdefault(l, []).append((n, s))
    for l, lst in sorted(by_layer.items()):
        t = f"model.layers.{l}.mlp.down_proj.weight"
        W = H.orig_tensor(t)
        for n, s in lst:
            W[:, n] = W[:, n] * s
        st = H.save_edit(t, W)
        print(f"{t}: neurons {[n for n, _ in lst]} scales {[s for _, s in lst]} -> {st['changed_after_bf16']} weights changed")
    es = H.edit_summary()
    print(f"candidate {name}: {es['tensors']} tensors, {es['changed_weights']} weights changed ({es['pct_changed']:.5f}%)")


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else FROZEN)
