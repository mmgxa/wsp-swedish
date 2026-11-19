"""Audit a run: rules, budget, reproducibility, holdout hygiene. Prints the results table.

    uv run --offline python sv/verify.py [--run RUN_DIR]
Exit code 1 if a hard check fails.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

HARD = {r"\.backward\(": "backward()", r"torch\.optim|\boptim\.": "optimizer", r"requires_grad": "requires_grad",
        r"autograd": "autograd", r"\bpeft\b|\blora\b": "LoRA/PEFT", r"from_pretrained|AutoModel|model_from_dir": "loads a model outside fwd()",
        r"_score_items|_generate_items|get_model\(|_teacher_forced|\.generate\(": "runs the model outside fwd()"}
SOFT = {r"lstsq|\bpinv\b|pinverse|linalg\.solve": "regression-like solve (check it is not fitting weights)",
        r"scipy\.optimize|\bcma\b|\.fit\(": "optimizer-like call (check)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    a = ap.parse_args()
    if a.run:
        os.environ["WSP_RUN_DIR"] = a.run
    import harness as H
    rd, cfg = H.run_dir(), H.config()
    bad, warn = [], []

    def ok(msg):
        print(f"  ok    {msg}")

    def fail(msg):
        bad.append(msg); print(f"  FAIL  {msg}")

    def wr(msg):
        warn.append(msg); print(f"  warn  {msg}")

    print(f"run {rd.name}")
    # 0 harness untouched
    for f, h in cfg["harness_sha256"].items():
        p = H.SV / f
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest() != h:
            fail(f"sv/{f} changed since the run was created")
    if not any("sv/" in b for b in bad):
        ok("harness files unchanged")

    # 1 forbidden APIs in agent code
    n_files = 0
    for p in sorted(rd.rglob("*.py")):
        n_files += 1
        txt = p.read_text(errors="ignore")
        for pat, name in HARD.items():
            if re.search(pat, txt):
                fail(f"{p.relative_to(rd)}: {name}")
        for pat, name in SOFT.items():
            if re.search(pat, txt):
                wr(f"{p.relative_to(rd)}: {name}")
        if "wsp_heldout_sv" in txt:
            fail(f"{p.relative_to(rd)}: touches the holdout path")
    for p in [rd / "notes.md", rd / "explanations.md"]:
        if p.exists() and "wsp_heldout_sv" in p.read_text():
            wr(f"{p.name} mentions the holdout path (check it was not read)")
    if not any(("py:" in b) for b in bad):
        ok(f"no forbidden API in {n_files} agent script(s)")

    # 2 pass log
    lines = [json.loads(x) for x in (rd / "forward_log.jsonl").read_text().splitlines() if x.strip()]
    passes = [x for x in lines if x.get("type") == "pass"]
    init = lines[0]
    if init.get("budget") != cfg["budget"]:
        fail("budget in config.json differs from the first log line")
    if [p["n"] for p in passes] != list(range(1, len(passes) + 1)):
        fail("pass numbering in forward_log.jsonl is not 1..n")
    if len(passes) > cfg["budget"]:
        fail(f"{len(passes)} passes > budget {cfg['budget']}")
    else:
        ok(f"{len(passes)}/{cfg['budget']} passes used")
    for p in passes:
        print(f"        pass {p['n']}: {p['kind']} {p['which']} {p['n_items']} items {p['tasks']} edit={p['edit_hash']} {p['secs']}s {p.get('note', '')[:60]}")

    # 3 edit
    ov = H._overlay_names()
    if set(ov) & H.FORBIDDEN_TENSORS:
        fail("overlay contains a forbidden tensor")
    ed = rd / "edited" / "model.safetensors"
    if not ov:
        wr("no edit saved (the original model is the submission)")
    else:
        es = H.edit_summary()
        (ok if es["pct_changed"] <= 1 else wr)(f"edit: {es['tensors']} tensors, {es['changed_weights']} weights ({es['pct_changed']:.4f}%) change after bf16 rounding")
        work = rd / "work" / "edit.py"
        if not work.exists():
            fail("work/edit.py is missing")
        else:
            with tempfile.TemporaryDirectory() as td:
                env = {**os.environ, "WSP_RUN_DIR": str(rd), "WSP_EDIT_DIR": td}
                r = subprocess.run([sys.executable, str(work)], cwd=H.ROOT, env=env, capture_output=True, text=True)
                new = Path(td) / "overlay.safetensors"
                if r.returncode != 0:
                    fail(f"work/edit.py failed: {r.stderr[-300:]}")
                elif not new.exists() or new.read_bytes() != H._overlay_path().read_bytes():
                    fail("work/edit.py does not reproduce edit/overlay.safetensors")
                else:
                    ok("work/edit.py reproduces the overlay byte for byte")
    if ed.exists():
        from safetensors import safe_open
        with safe_open(str(ed), "pt") as e, safe_open(str(H.MODEL_DIR / "model.safetensors"), "pt") as o:
            same = all((e.get_tensor(k) == o.get_tensor(k)).all().item() for k in H.FORBIDDEN_TENSORS if k in o.keys())
        (ok if same else fail)("embed_tokens and final norm identical in edited/")
        if hashlib.sha256(ed.read_bytes()).hexdigest() != json.loads((rd / "edit_stats.json").read_text())["model_sha256"]:
            wr("edited/ is stale: it was built from an earlier edit than the current overlay")

    # 4 holdout hygiene
    fl = rd / "final_eval_log.jsonl"
    if fl.exists():
        ev = [json.loads(x) for x in fl.read_text().splitlines() if x.strip()]
        # An entry with overlay_hash "none" is a baseline (original vs original): it cannot select between edits.
        base = [e for e in ev if e["overlay_hash"] == "none"]
        cand = [e for e in ev if e["overlay_hash"] != "none"]
        hashes = {e["overlay_hash"] for e in cand}
        if len(hashes) > 1:
            fail(f"final_eval ran on {len(hashes)} distinct edits: the holdout was used to choose between edits")
        else:
            ok(f"final_eval ran {len(cand)} time(s) on {len(hashes)} distinct edit(s)")
        if base:
            wr(f"final_eval ran {len(base)} time(s) with no edit saved (baseline, original vs original); "
               "it shows the holdout numbers of the original model, check the agent did not read results.json then")
        if len(cand) > 1:
            wr(f"final_eval ran {len(cand)} times on the same edit (the agent rule is once per frozen candidate)")
        if hashes and H.edit_hash() not in hashes and ov:
            wr("final_eval was run on an edit that is no longer the current one")
    else:
        wr("final_eval has not been run")

    # 5 results
    rp = rd / "results.json"
    if rp.exists():
        from final_eval import print_table
        print()
        print_table(json.loads(rp.read_text()))
    print(f"\n{len(bad)} failure(s), {len(warn)} warning(s)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
