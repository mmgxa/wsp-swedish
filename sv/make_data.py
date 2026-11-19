"""Export a small, configurable Swedish evaluation set from EuroEval.

Run with the evaluation environment (it has the `euroeval` package, which is the
sanctioned way to read the EuroEval datasets):

    .venv-eval/bin/python sv/make_data.py --n-dev 100 --n-holdout 200

DEV items come from the EuroEval *train* split and go to sv/data/<name>/dev/. The editor
agent may read and use them (through the budgeted fwd()).
HOLDOUT items come from the EuroEval *test* split and go OUTSIDE the workspace, to
$WSP_HELDOUT_ROOT/<name>/ (default /opt/dlami/nvme/work/wsp_heldout_sv). Only final_eval.py reads them.

Item formats (one JSON object per line):
  mc : {"id","task","kind":"mc","prompt","choices":[...],"gold":int}
  lm : {"id","task","kind":"lm","text"}
  gen: {"id","task","kind":"gen","prompt"}

Tasks (all Swedish unless noted):
  lm        pooled Swedish text (ScaLA-sv correct sentences, SweReC, SweDN, MultiWikiQA-sv)
  lm_en     English text (English side of ScandiQA-sv): retention check, not a target
  scala     ScaLA-sv, grammatical acceptability (ja/nej)
  swerec    SweReC, sentiment (positiv/neutral/negativ)
  hellaswag HellaSwag-sv (a-d)
  arc       ARC-sv (a-d)
  mmlu      MMLU-sv (a-d)
  belebele  Belebele-sv (a-d)
  gen       Swedish questions (MultiWikiQA-sv, ScandiQA-sv), answered by greedy generation
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ALL_TASKS = ["lm", "lm_en", "scala", "swerec", "hellaswag", "arc", "mmlu", "belebele", "gen"]
MAX_TEXT_CHARS = 700  # cap for lm / long prompts so one item stays short
MAX_PROMPT_CHARS = 1800

# EuroEval's Swedish zero-shot instruction prompts (euroeval/prompt_templates, v18.4)
MC_PROMPT = "Fråga: {text}\n\nBesvara följande fråga med a, b, c eller d, och inget annat."
LA_PROMPT = "Mening: {text}\n\nBestäm om meningen är grammatiskt korrekt eller inte. Svara med ja eller nej, och inget annat."
SENT_PROMPT = "Dokument: {text}\n\nKlassificera känslan i dokumentet. Svara med positiv, neutral eller negativ, och inget annat."


def clip(text: str, n: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= n:
        return text
    cut = text[:n]
    m = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
    return cut[: m + 1] if m > n // 2 else cut


def load(cfg_name: str, cache_dir: str):
    from euroeval.data_loading import load_raw_data
    from euroeval.dataset_configs import swedish as S

    return load_raw_data(getattr(S, cfg_name), cache_dir=cache_dir, api_key=None)


class Builder:
    def __init__(self, split: str, rng: random.Random, cache_dir: str):
        self.split, self.rng, self.cache = split, rng, cache_dir
        self._ds: dict[str, object] = {}

    def ds(self, cfg: str):
        if cfg not in self._ds:
            self._ds[cfg] = load(cfg, self.cache)[self.split]
        return self._ds[cfg]

    def sample(self, cfg: str, n: int, pred=None) -> list[dict]:
        d = self.ds(cfg)
        idx = [i for i in range(len(d)) if pred is None or pred(d[i])]
        self.rng.shuffle(idx)
        return [d[i] | {"_idx": i} for i in idx[:n]]

    # ---- tasks ----
    def mc(self, task: str, cfg: str, prompt_fmt: str, labels: list[str], n: int, answers: list[str] | None = None):
        out = []
        for r in self.sample(cfg, n):
            prompt = prompt_fmt.format(text=clip_mc(r["text"]))
            choices = answers or labels
            out.append(dict(id=f"{task}-{r['_idx']}", task=task, kind="mc", prompt=prompt, choices=choices,
                            gold=labels.index(r["label"])))
        return out

    def build(self, task: str, n: int) -> list[dict]:
        if task == "scala":
            return self.mc(task, "SCALA_SV_CONFIG", LA_PROMPT, ["correct", "incorrect"], n, ["ja", "nej"])
        if task == "swerec":
            return self.mc(task, "SWEREC_CONFIG", SENT_PROMPT, ["positive", "neutral", "negative"], n,
                           ["positiv", "neutral", "negativ"])
        cfg = {"hellaswag": "HELLASWAG_SV_CONFIG", "arc": "ARC_SV_CONFIG", "mmlu": "MMLU_SV_CONFIG",
               "belebele": "BELEBELE_SV_CONFIG"}.get(task)
        if cfg:
            return self.mc(task, cfg, MC_PROMPT, ["a", "b", "c", "d"], n)
        if task == "lm":
            parts = [("SCALA_SV_CONFIG", lambda r: r["text"], lambda r: r["label"] == "correct"),
                     ("SWEREC_CONFIG", lambda r: r["text"], None),
                     ("SWEDN_CONFIG", lambda r: r["text"], None),
                     ("MULTI_WIKI_QA_SV_CONFIG", lambda r: r["context"], None)]
            per = [n // len(parts) + (1 if i < n % len(parts) else 0) for i in range(len(parts))]
            out = []
            for (cfg, get, pred), k in zip(parts, per):
                for r in self.sample(cfg, k, pred):
                    txt = clip(get(r), MAX_TEXT_CHARS)
                    if len(txt) >= 30:
                        out.append(dict(id=f"lm-{cfg.split('_CONFIG')[0].lower()}-{r['_idx']}", task="lm", kind="lm", text=txt))
            return out
        if task == "lm_en":
            return [dict(id=f"lm_en-{r['_idx']}", task="lm_en", kind="lm", text=clip(r["context_en"], MAX_TEXT_CHARS))
                    for r in self.sample("SCANDIQA_SV_CONFIG", n)]
        if task == "gen":
            a = self.sample("MULTI_WIKI_QA_SV_CONFIG", n // 2 + n % 2)
            b = self.sample("SCANDIQA_SV_CONFIG", n // 2)
            return [dict(id=f"gen-{i}", task="gen", kind="gen", prompt=r["question"].strip())
                    for i, r in enumerate(a + b)]
        raise SystemExit(f"unknown task {task}")


def clip_mc(text: str) -> str:
    return text if len(text) <= MAX_PROMPT_CHARS else text[:MAX_PROMPT_CHARS].rsplit(" ", 1)[0]


def write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="default", help="data set name (directory name)")
    ap.add_argument("--n-dev", type=int, default=100, help="DEV items per task (from the EuroEval train split)")
    ap.add_argument("--n-holdout", type=int, default=200, help="HOLDOUT items per task (from the EuroEval test split)")
    ap.add_argument("--tasks", nargs="+", default=ALL_TASKS, choices=ALL_TASKS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-dir", default=str(ROOT.parent / "wsp_heldout_sv" / ".euroeval_cache"), help="raw EuroEval cache; kept outside the workspace because it holds the test splits")
    ap.add_argument("--heldout-root", default=os.environ.get("WSP_HELDOUT_ROOT", str(ROOT.parent / "wsp_heldout_sv")))
    a = ap.parse_args()

    dev_dir = ROOT / "sv" / "data" / a.name / "dev"
    held_dir = Path(a.heldout_root) / a.name
    meta = {"name": a.name, "seed": a.seed, "tasks": a.tasks, "n_dev": a.n_dev, "n_holdout": a.n_holdout,
            "source": "EuroEval Swedish datasets, loaded through the euroeval package",
            "dev_split": "train", "holdout_split": "test", "heldout_dir": str(held_dir), "counts": {}}
    for split_name, split, n, out_dir in [("dev", "train", a.n_dev, dev_dir), ("holdout", "test", a.n_holdout, held_dir)]:
        b = Builder(split, random.Random(f"{a.seed}-{split}"), a.cache_dir)
        for t in a.tasks:
            rows = b.build(t, n)
            write_jsonl(out_dir / f"{t}.jsonl", rows)
            meta["counts"][f"{split_name}/{t}"] = len(rows)
            print(f"{split_name:8s} {t:10s} {len(rows)}")
    meta_public = {k: v for k, v in meta.items() if k != "heldout_dir"}
    (ROOT / "sv" / "data" / a.name / "meta.json").write_text(json.dumps(meta_public, indent=1))
    held_dir.mkdir(parents=True, exist_ok=True)
    (held_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    print("dev    ->", dev_dir)
    print("holdout ->", held_dir, "(outside the workspace; do not read from the editing run)")


if __name__ == "__main__":
    main()
