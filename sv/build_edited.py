"""Build the edited model: original weights + the edit overlay -> <run>/edited/ (a loadable copy).

Deterministic. The overlay tensors are cast to bf16 (the model's storage format), so rounding is included.
    uv run python sv/build_edited.py [--out DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402


def build(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    for f in H.MODEL_DIR.iterdir():
        if f.is_file() and f.name != "model.safetensors":
            shutil.copy2(f, out / f.name)
    sd = load_file(str(H.MODEL_DIR / "model.safetensors"))
    ov = H._overlay()
    stats = {"tensors": {}, "changed_weights": 0}
    total = sum(t.numel() for t in sd.values())
    for name, t in ov.items():
        if name in H.FORBIDDEN_TENSORS:
            raise SystemExit(f"{name} may not be edited")
        new = t.to(torch.bfloat16)
        d = (new.float() - sd[name].float())
        ch = int((new != sd[name]).sum())
        stats["tensors"][name] = {"numel": new.numel(), "changed": ch, "max_abs": float(d.abs().max()),
                                  "mean_abs_over_changed": float(d.abs().sum() / max(ch, 1))}
        stats["changed_weights"] += ch
        sd[name] = new.contiguous()
    stats.update(total_weights=total, pct_changed=100 * stats["changed_weights"] / total, n_tensors=len(ov),
                 overlay_hash=H.edit_hash())
    save_file({k: sd[k] for k in sorted(sd)}, str(out / "model.safetensors"), metadata={"format": "pt"})
    stats["model_sha256"] = hashlib.sha256((out / "model.safetensors").read_bytes()).hexdigest()
    (out.parent / "edit_stats.json").write_text(json.dumps(stats, indent=1))
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out) if a.out else H.run_dir() / "edited"
    s = build(out)
    print(f"tensors touched: {s['n_tensors']}; weights changed: {s['changed_weights']} of {s['total_weights']} "
          f"({s['pct_changed']:.4f}%); sha256 {s['model_sha256'][:16]}")
    for k, v in s["tensors"].items():
        print(f"  {k}: changed {v['changed']}/{v['numel']} max|d|={v['max_abs']:.4g}")
