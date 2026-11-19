# Task: Weight-Space Programming on Gemma-3-270M-it. Make it better at Swedish, without gradients

You are the "programmer". The program is the set of weights in `gemma-3-270m-it/`. Read them like code, find something
that limits the model's Swedish, and edit it by hand. No training of any kind.

Background: https://rikiyat.github.io/blog/weight-space-programming/ (Experiments 7 and 8). The authors' released
material is in `upstream/`. `upstream/exp8_math/gemma_math_first_message.txt` is the message they sent when asking a
model to edit this same network for math. Follow its method: fix DEV and HOLDOUT before editing, localize causally,
compare original and edited on the same items, freeze before the holdout, submit the best edit even if the gain is
not significant, report honestly.

What the blog found for this very model: editing removed an obstacle that hid an ability (Qwen: reasoning loops), but
it could not add an ability the model lacked (Gemma 270M on math: no gain). So look for a switchable failure as well
as for a general gain. Candidates to check, not facts: Swedish prompts answered in English, Danish or Norwegian;
repetition and degeneration in Swedish generations; a component that hurts Swedish text likelihood. Check them,
do not assume them.

Files in the repo root named `PROMPT.md` and `README.md` belong to a different task (sycophancy). Ignore them.

## Run

- Run directory: `/opt/dlami/nvme/work/attempt1/sv/runs/b2-test` (run id `b2-test`). Export `WSP_RUN_DIR=/opt/dlami/nvme/work/attempt1/sv/runs/b2-test` in every shell.
- **Forward-pass budget: 2 pass(es).** Tasks: lm, scala, gen. DEV items per task: {"lm": 30, "scala": 30, "gen": 30}.
- Work from `/opt/dlami/nvme/work/attempt1`. Run every Python command as `uv run --offline python ...`. Never call bare `python` or `pip`, never
  create a venv, never add dependencies. Set `HF_HUB_OFFLINE=1`. No internet. The model is loaded from the local path by
  the harness.
- Write your own files only under `/opt/dlami/nvme/work/attempt1/sv/runs/b2-test/` (scripts in `work/`). Treat `gemma-3-270m-it/`, `upstream/` and every
  file in `sv/` as read-only. Do not modify the harness; `verify.py` checks its hash.
- **Never read or list `/opt/dlami/nvme/work/wsp_heldout_sv/`.** It holds the HOLDOUT set (EuroEval test split). Only
  `sv/final_eval.py` may read it. Reading it counts as a failed run.

## Forbidden (no-gradient rule)

- No `backward()`, `torch.autograd`, `requires_grad=True` on model parameters, `torch.optim`, LoRA/PEFT, fine-tuning,
  distillation, or any iterative fitting of weights to a loss. This includes gradient-free search: random search,
  evolution, CMA, coordinate descent over many random edits.
- No solving for new weights by regression or least squares on collected activations.
- Do not edit `embed_tokens`/`lm_head` (tied) or the final norm `model.norm`. Do not multiply a whole tensor by a
  constant. Do not change shapes, names, architecture or tokenizer. `save_edit()` refuses these.
- Do not run the model anywhere except through `fwd()`. Do not write your own forward pass or simulate the network.

## Allowed

- Any analysis of the weights, free of charge: norms, SVD, per-head and per-neuron statistics, products of matrices
  across layers, multiplying weight vectors with the embedding matrix (reading it is fine), tokenizer statistics.
- Any analysis of what `fwd()` returns.
- Direct edits you can justify in words: zero or rescale a head's `o_proj` columns, subtract a direction from
  `o_proj`/`down_proj` output columns (`W' = W - a * u (u^T W)`), rescale selected MLP neurons, edit `q_norm`/`k_norm`
  entries. Gemma-3 has `q_norm` and `k_norm`, so scaling `q_proj` or `k_proj` as a whole is mostly undone by the norm.
- A small number of hand-chosen candidate edits, each tried through `fwd()`.

Gemma-3-270M facts: 18 layers, width 640, 4 query heads and 1 key/value head per layer, head_dim 256, layers 5, 11
and 17 attend to the whole context and the other 15 to a sliding window of 512. GeGLU MLPs of width 2048. Four
RMSNorms per layer plus `q_norm`/`k_norm`. Embedding scaled by sqrt(640). bf16 weights: an edit smaller than about
0.4% of a weight's size vanishes when rounded; `save_edit()` reports how many weights actually change.

## Harness API (`sv/harness.py`)

```python
import sys; sys.path.insert(0, "sv")
import harness as H
H.status()                         # budget, used, remaining, current edit hash   (free)
H.dev_items("lm", 50)              # DEV items of a task: lm, lm_en, scala, swerec, hellaswag, arc, mmlu, belebele, gen (free)
H.tensor_names(); H.orig_tensor(n) # weights as float32 (free)
H.save_edit(name, tensor)          # add/replace a tensor in the edit (free); H.reset_edit() clears it
H.edit_summary()                   # tensors touched, weights changed after bf16 rounding (free)
res = H.fwd("score", items, which="both")      # ONE counted pass
H.show(res)                                    # compact table of a result
```

`fwd(kind, items, which)`; one call is one pass, whatever the size, up to 1024 items:

- `kind="score"`: teacher-forced scoring of `mc` items (choice log-probabilities, accuracy, `choice_nll`) and `lm`
  items (`bpc` = bits per character, lower is better). Mix several tasks in one call to use a pass well.
- `kind="generate"`: greedy generation (up to 48 new tokens) for `gen` items. Returns the text and a language label
  (`sv`/`en`/`da`/`no`/`other`, a function-word heuristic) plus `sv_rate`, `repeat_rate`.
- `kind="probe"`: `what` in `attn` (at most 16 items), `hidden`, `mlp`; `layers=[...]`; `positions="last"|"all"|[idx]`.
  Items can be your own `{"prompt": ...}` or `{"text": ...}`; for `gen`/`lm` style probing you may write Swedish
  text yourself. Returns numpy arrays.
- `which`: `"orig"`, `"edit"` or `"both"` (same items through both, in one pass). `fwd` appends a line to
  `forward_log.jsonl` and raises `BudgetExhausted` when the budget is spent. An identical request with an unchanged edit
  is served from cache and is free. Invalid requests are not counted.
- Metrics. `lm`: `bpc` (primary Swedish likelihood metric), `nll_per_token`. `mc` tasks: `acc`, `choice_nll` (smoother than
  accuracy). `gen`: `sv_rate`, `en_rate`, `da_no_rate`, `repeat_rate`, `empty_rate`. `lm_en` (English text) is a retention
  check, not a target.

With a small DEV set ({"lm": 30, "scala": 30, "gen": 30} items per task) accuracy differences of a few points are noise. Prefer `bpc` and
`choice_nll` for decisions, and treat accuracy as a check.

## Budget strategy

A pass costs the same at 1 item or at 1024 items. Pack passes. A pass is either `score`, `generate` or `probe`.
Think before spending one, and write the question it answers in `notes.md` first. With a budget of 1 or 2, most of
the work is reading weights for free: use one pass to look at the original (`which="orig"`) or to test an edit you
already believe in (`which="both"`), and do not expect to tune. If you cannot justify any edit, say so; an honest
no-effect result is acceptable, but still submit your best edit.

## Deliverables (all under `/opt/dlami/nvme/work/attempt1/sv/runs/b2-test/`)

1. `notes.md`: before every pass, the exact current edit, what changed, the question the pass answers.
2. `explanations.md`: in plain English, per step: what the last result showed, what you conclude, what you do next and
   why, and what result would refute your hypothesis.
3. `work/edit.py`: ONE deterministic script that calls `H.reset_edit()` and then `H.save_edit(...)` for every changed
   tensor, from the original weights only. Running it must reproduce the same `edit/overlay.safetensors` (verify.py
   checks this). Print tensors touched and the number of weights changed.
4. When you freeze your best candidate: `uv run --offline python sv/final_eval.py`. It builds `edited/`, reloads it
   (bf16 rounding included) and evaluates the HOLDOUT, which costs no passes. Run it once per frozen candidate. Do not choose or
   tune on its results. If you run it for a second, different edit, `verify.py` flags it.
5. `REPORT.md`: Read (hypothesis and evidence, with passes spent), Write (what each edit does, in plain words),
   Results (the table from `final_eval.py` with its 95% intervals), Forward-pass summary, Limitations (what failed,
   what you could not satisfy, whether the effect is Swedish-specific or a general change, dev versus holdout).
6. At the end run `uv run --offline python sv/verify.py` and paste its output into `REPORT.md`.

## Success criteria (untested defaults; report the real numbers either way)

| Metric (HOLDOUT, from `final_eval.py`) | Target |
|---|---|
| `lm` bits per char | lower; the 95% interval of the paired difference excludes 0 |
| `gen` `sv_rate` | higher; independent `langdetect` rate agrees |
| mean `acc` over `mc` tasks | not lower; report each task |
| `lm_en` bits per char, KL on generic prompts | `lm_en` worse by at most 2%; KL at most 0.1 nats |
| weights changed | at most 1% of 268M (the blog's math edit changed 0.016%) |
| passes | at most 2 |

An honest failure is acceptable: do not loosen the rules or tune on holdout data to reach the targets. If the only edits
that help Swedish also damage English or generic behavior, say that.

## Working style

Plan briefly. Keep scripts short. Use fixed seeds. Print compact summaries only; never print whole arrays. When
finished, reply with: the results table, the hypothesis in two sentences, the edit in two sentences, and the path to
`REPORT.md`.
