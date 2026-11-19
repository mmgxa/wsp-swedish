"""Final evaluation on the HOLDOUT set (EuroEval test split). Not budgeted. Run once per frozen candidate.

    uv run python sv/final_eval.py [--n-holdout 200]

Builds <run>/edited/ from the edit overlay, reloads that copy (so bf16 rounding counts), and compares it with the
original on every task: accuracy / bits per character / language rates with 95% intervals (Wilson for rates,
paired bootstrap for differences), plus KL(orig || edited) on generic prompts. Appends to final_eval_log.jsonl;
verify.py flags more than one evaluated edit (that would be selecting on the holdout).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402
from build_edited import build  # noqa: E402


def wilson(k: float, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def boot(stat, n: int, B: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    v = [stat(rng.integers(0, n, n)) for _ in range(B)]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


@torch.no_grad()
def kl_last(mo, me, prompts: list[str]) -> list[float]:
    out = []
    for p in prompts:
        ids = torch.tensor([H._ids(H._chat(p))], device=H.DEVICE)
        lp = []
        for m in (mo, me):
            h = m.model(input_ids=ids).last_hidden_state[0, -1:]
            lg = m.lm_head(h).float()
            cap = getattr(m.config, "final_logit_softcapping", None)
            lg = torch.tanh(lg / cap) * cap if cap else lg
            lp.append(torch.log_softmax(lg, -1)[0])
        out.append(float((lp[0].exp() * (lp[0] - lp[1])).sum()))
    return out


@torch.no_grad()
def kl_text(mo, me, texts: list[str]) -> list[float]:
    out = []
    tok = H.tokenizer()
    for t in texts:
        ids = torch.tensor([[tok.bos_token_id] + H._ids(t)], device=H.DEVICE)
        lp = []
        for m in (mo, me):
            h = m.model(input_ids=ids).last_hidden_state[0, :-1]
            lg = m.lm_head(h).float()
            cap = getattr(m.config, "final_logit_softcapping", None)
            lg = torch.tanh(lg / cap) * cap if cap else lg
            lp.append(torch.log_softmax(lg, -1))
        out.append(float((lp[0].exp() * (lp[0] - lp[1])).sum(-1).mean()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-holdout", type=int, default=None, help="use only the first N items per task")
    a = ap.parse_args()
    cfg = H.config()
    rd = H.run_dir()
    held = Path(os.environ.get("WSP_HELDOUT_ROOT", H.ROOT.parent / "wsp_heldout_sv")) / cfg["data"]

    print("building edited model ...")
    stats = build(rd / "edited")
    mo = H.model_from_dir(str(H.MODEL_DIR))
    me = H.model_from_dir(str(rd / "edited"))

    results = {"run": rd.name, "budget": cfg["budget"], "passes_used": H.passes_used(), "edit": {
        "pct_changed": stats["pct_changed"], "changed_weights": stats["changed_weights"], "tensors": stats["n_tensors"],
        "overlay_hash": stats["overlay_hash"], "model_sha256": stats["model_sha256"]}, "tasks": {}}
    gen_texts = {"orig": [], "edit": []}
    examples = []
    for task in cfg["tasks"]:
        items = H.load_jsonl(held / f"{task}.jsonl", a.n_holdout)
        t0 = time.time()
        if items[0]["kind"] == "gen":
            ro = H._generate_items(mo, items, cfg["max_new_tokens"])
            re_ = H._generate_items(me, items, cfg["max_new_tokens"])
            gen_texts["orig"] = [r["text"] for r in ro]; gen_texts["edit"] = [r["text"] for r in re_]
            examples = [{"prompt": it["prompt"], "orig": x["text"], "edit": y["text"]} for it, x, y in zip(items[:10], ro, re_)]
        else:
            ro = H._score_items(mo, items)
            re_ = H._score_items(me, items)
        so, se = H.summarize(items, ro)[task], H.summarize(items, re_)[task]
        n = len(items)
        row = {"n": n, "orig": so, "edit": se}
        kind = items[0]["kind"]
        if kind == "mc":
            co = np.array([r["correct"] for r in ro], float); ce = np.array([r["correct"] for r in re_], float)
            row["acc_ci"] = {"orig": wilson(co.sum(), n), "edit": wilson(ce.sum(), n)}
            row["delta_acc"] = float(ce.mean() - co.mean())
            row["delta_acc_ci"] = boot(lambda i: ce[i].mean() - co[i].mean(), n)
            no = np.array([r["choice_nll"] for r in ro]); ne = np.array([r["choice_nll"] for r in re_])
            row["delta_choice_nll"] = float(ne.mean() - no.mean())
            row["delta_choice_nll_ci"] = boot(lambda i: ne[i].mean() - no[i].mean(), n)
        elif kind == "lm":
            no = np.array([r["nll"] for r in ro]); ne = np.array([r["nll"] for r in re_]); nc = np.array([r["n_chars"] for r in ro], float)
            f = lambda x, i: x[i].sum() / nc[i].sum() / math.log(2)
            row["delta_bpc"] = float(f(ne, np.arange(n)) - f(no, np.arange(n)))
            row["delta_bpc_ci"] = boot(lambda i: f(ne, i) - f(no, i), n)
            row["bpc_ci"] = {"orig": boot(lambda i: f(no, i), n), "edit": boot(lambda i: f(ne, i), n)}
        else:
            for k in ("sv_rate", "repeat_rate"):
                vo = np.array([(r["lang"] == "sv") if k == "sv_rate" else H._rep(r["text"]) for r in ro], float)
                ve = np.array([(r["lang"] == "sv") if k == "sv_rate" else H._rep(r["text"]) for r in re_], float)
                row[f"{k}_ci"] = {"orig": wilson(vo.sum(), n), "edit": wilson(ve.sum(), n)}
                row[f"delta_{k}"] = float(ve.mean() - vo.mean())
                row[f"delta_{k}_ci"] = boot(lambda i, vo=vo, ve=ve: ve[i].mean() - vo[i].mean(), n)
        results["tasks"][task] = row
        print(f"  {task:10s} n={n:4d} ({time.time() - t0:.0f}s)")

    # drift
    gen_items = H.load_jsonl(held / "gen.jsonl", a.n_holdout) if "gen" in cfg["tasks"] else []
    if gen_items:
        results["kl_generic_prompts_nats"] = float(np.mean(kl_last(mo, me, [g["prompt"] for g in gen_items])))
    if "lm_en" in cfg["tasks"]:
        results["kl_english_text_nats_per_token"] = float(np.mean(kl_text(mo, me, [x["text"] for x in H.load_jsonl(held / "lm_en.jsonl", 50)])))
    results["generation_examples"] = examples

    # independent language check with langdetect (evaluation env), if present
    ev = H.ROOT / ".venv-eval" / "bin" / "python"
    if ev.exists() and gen_texts["orig"]:
        tmp = held / "_gen_tmp.json"; tmp.write_text(json.dumps(gen_texts))
        r = subprocess.run([str(ev), str(H.SV / "langdetect_rates.py"), str(tmp), str(held / "_ld.json")], capture_output=True, text=True)
        if r.returncode == 0:
            results["langdetect"] = json.loads((held / "_ld.json").read_text())
        tmp.unlink(missing_ok=True)

    (rd / "results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
    with (rd / "final_eval_log.jsonl").open("a") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "overlay_hash": stats["overlay_hash"],
                            "n_holdout": a.n_holdout}) + "\n")
    print_table(results)


def print_table(r: dict):
    print(f"\n{r['run']}: {r['passes_used']}/{r['budget']} passes; {r['edit']['pct_changed']:.4f}% of weights changed "
          f"({r['edit']['tensors']} tensors)")
    print(f"{'task':10s} {'n':>4s}  {'metric':12s} {'orig':>8s} {'edit':>8s} {'delta':>8s}  95% CI of delta")
    for t, row in r["tasks"].items():
        o, e = row["orig"], row["edit"]
        if "delta_acc" in row:
            print(f"{t:10s} {row['n']:4d}  {'acc':12s} {o['acc']:8.3f} {e['acc']:8.3f} {row['delta_acc']:+8.3f}  [{row['delta_acc_ci'][0]:+.3f}, {row['delta_acc_ci'][1]:+.3f}]")
            print(f"{'':10s} {'':4s}  {'choice_nll':12s} {o['choice_nll']:8.3f} {e['choice_nll']:8.3f} {row['delta_choice_nll']:+8.3f}  [{row['delta_choice_nll_ci'][0]:+.3f}, {row['delta_choice_nll_ci'][1]:+.3f}]")
        elif "delta_bpc" in row:
            print(f"{t:10s} {row['n']:4d}  {'bits/char':12s} {o['bpc']:8.3f} {e['bpc']:8.3f} {row['delta_bpc']:+8.3f}  [{row['delta_bpc_ci'][0]:+.3f}, {row['delta_bpc_ci'][1]:+.3f}]")
        else:
            for k in ("sv_rate", "repeat_rate"):
                print(f"{t:10s} {row['n']:4d}  {k:12s} {o[k]:8.3f} {e[k]:8.3f} {row['delta_' + k]:+8.3f}  [{row['delta_' + k + '_ci'][0]:+.3f}, {row['delta_' + k + '_ci'][1]:+.3f}]")
    for k in ("kl_generic_prompts_nats", "kl_english_text_nats_per_token"):
        if k in r:
            print(f"{k}: {r[k]:.4f}")
    if "langdetect" in r:
        print("langdetect (independent):", {s: {k: round(v, 3) for k, v in d.items() if k != 'n'} for s, d in r["langdetect"].items()})


if __name__ == "__main__":
    main()
