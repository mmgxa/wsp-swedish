# Weight-space programming for Swedish (Gemma-3-270M-it)

An LLM agent (Kiro CLI, Claude Opus 5.5) edits the weights of `gemma-3-270m-it` by hand, with no gradients, to improve
its Swedish. The model is run only through a counted `fwd()`; the edit is judged on a held-out EuroEval subset.

## Quick start

```bash
cd /opt/dlami/nvme/work/attempt1
export HF_HUB_OFFLINE=1

# 1. Data (once). Evaluation env has the euroeval package. n = items PER TASK.
.venv-eval/bin/python sv/make_data.py --name default --n-dev 100 --n-holdout 200      # default
.venv-eval/bin/python sv/make_data.py --name tiny --n-dev 30 --n-holdout 50 --tasks lm scala gen   # smaller

# 2. A run. --budget is the number of counted forward passes.
uv run --offline python sv/init_run.py --budget 2 --data default --run-id b2-test

# 3. The agent
sv/run_agent.sh b2-test              # MODEL=claude-opus-5.5 EFFORT=high by default

# 4. Audit and results
uv run --offline python sv/verify.py --run sv/runs/b2-test
cat sv/runs/b2-test/REPORT.md
```
Repeat step 2-4 with `--budget 10`, `25`, `50`, same data, to see how results scale.

## Where things are

| Path | What |
|---|---|
| `sv/data/<name>/dev/*.jsonl` | DEV items (EuroEval **train** split). The agent reads them and may use them in `fwd()`. |
| `/opt/dlami/nvme/work/wsp_heldout_sv/<name>/` | HOLDOUT items (EuroEval **test** split), plus the raw EuroEval cache. Outside the workspace. Only `final_eval.py` reads it. |
| `sv/harness.py` | `fwd()` (counted), `save_edit()`, scoring, language heuristic. |
| `sv/runs/<id>/` | `config.json`, `PROMPT.md` (rendered), `forward_log.jsonl`, `notes.md`, `explanations.md`, `work/edit.py`, `edit/overlay.safetensors`, `edited/`, `results.json`, `REPORT.md`, `agent.log` |

## Tasks (all from EuroEval, Swedish)

`lm` Swedish text bits/char (pooled ScaLA-sv, SweReC, SweDN, MultiWikiQA-sv) | `lm_en` English text (retention check) |
`scala` acceptability | `swerec` sentiment | `hellaswag`, `arc`, `mmlu`, `belebele` multiple choice (zero-shot, EuroEval's
Swedish instruction prompt, scored by choice log-probability) | `gen` Swedish questions, greedy answer, language label.

## Why a custom scorer and not the `euroeval` CLI

EuroEval's Swedish datasets are private on the Hugging Face Hub; the `euroeval` package is the supported way to read
them, so `make_data.py` loads them through its loader (in `.venv-eval`, separate from the editing env). Running local
generative models in EuroEval needs vLLM, which is not installed. `final_eval.py` therefore re-implements the scoring
on the exported items. Its numbers are comparable between orig and edited model, **not** to the EuroEval leaderboard.

## Integrity (honor system plus audit)

The agent runs as your user, so nothing is sandboxed. `verify.py` checks: harness files unchanged, no forbidden API in
agent scripts, pass count within budget, `work/edit.py` reproduces the overlay byte for byte, embeddings and final norm
unchanged, and that `final_eval.py` was run on one edit only. It cannot prove the agent never read the holdout directory.
