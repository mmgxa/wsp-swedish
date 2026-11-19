"""Create a run directory: budget, task list, rendered agent prompt, sealed harness hashes.

    uv run python sv/init_run.py --budget 2
    uv run python sv/init_run.py --budget 10 --data default --tasks lm scala gen --run-id b10-swedish
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SV = Path(__file__).resolve().parent
ROOT = SV.parent
GUARDED = ["harness.py", "build_edited.py", "final_eval.py", "init_run.py", "make_data.py", "langdetect_rates.py", "verify.py"]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, required=True, help="number of counted forward passes")
    ap.add_argument("--data", default="default", help="data set name from sv/make_data.py")
    ap.add_argument("--tasks", nargs="+", default=None, help="subset of the data set's tasks (default: all)")
    ap.add_argument("--max-items-per-pass", type=int, default=1024)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--run-id", default=None)
    a = ap.parse_args()
    if a.budget < 1:
        sys.exit("--budget must be >= 1")

    meta_p = SV / "data" / a.data / "meta.json"
    if not meta_p.exists():
        sys.exit(f"no data set {a.data}: run .venv-eval/bin/python sv/make_data.py --name {a.data} first")
    meta = json.loads(meta_p.read_text())
    tasks = a.tasks or meta["tasks"]
    bad = [t for t in tasks if t not in meta["tasks"]]
    if bad:
        sys.exit(f"tasks not in data set: {bad}")
    rid = a.run_id or f"b{a.budget}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    rd = SV / "runs" / rid
    if rd.exists():
        sys.exit(f"{rd} exists")
    (rd / "work").mkdir(parents=True)
    cfg = {"run": rid, "budget": a.budget, "data": a.data, "tasks": tasks, "max_items_per_pass": a.max_items_per_pass,
           "max_new_tokens": a.max_new_tokens, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "n_dev_per_task": meta["n_dev"], "harness_sha256": {f: sha(SV / f) for f in GUARDED if (SV / f).exists()}}
    (rd / "config.json").write_text(json.dumps(cfg, indent=1))
    (rd / "forward_log.jsonl").write_text(json.dumps({"type": "init", "budget": a.budget, "run": rid, "created": cfg["created"]}) + "\n")
    (rd / "notes.md").write_text("# Notes (write the exact current edit and what changed BEFORE every pass)\n")
    (rd / "explanations.md").write_text("# Explanations (plain English, one entry per step)\n")

    counts = {t: meta["counts"].get(f"dev/{t}") for t in tasks}
    prompt = (SV / "AGENT_PROMPT.md").read_text()
    for k, v in {"@@RUN_DIR@@": str(rd), "@@RUN_ID@@": rid, "@@BUDGET@@": str(a.budget), "@@TASKS@@": ", ".join(tasks),
                 "@@MAX_ITEMS@@": str(a.max_items_per_pass), "@@MAX_NEW@@": str(a.max_new_tokens),
                 "@@DEV_COUNTS@@": json.dumps(counts), "@@DATA@@": a.data, "@@ROOT@@": str(ROOT)}.items():
        prompt = prompt.replace(k, v)
    (rd / "PROMPT.md").write_text(prompt)
    (SV / "runs" / "CURRENT").write_text(str(rd))
    print(f"run {rid}: budget {a.budget}, tasks {tasks}\n  dir    {rd}\n  prompt {rd / 'PROMPT.md'}\n  next   sv/run_agent.sh {rid}")


if __name__ == "__main__":
    main()
