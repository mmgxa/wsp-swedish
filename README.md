# Weight-space programming for Swedish (Gemma-3-270M-it)

Can an LLM agent improve the Swedish of `google/gemma-3-270m-it` by editing its weights by hand, with no gradient
training and no LoRA? An agent (Kiro CLI with Claude Opus 5.5) may read the model only through a counted forward-pass
function. Its edit is scored on a held-out EuroEval Swedish subset. The method follows
[Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/). Three runs used budgets
of 2, 5 and 10 forward passes.

## Where to read

| File | Content |
|---|---|
| `REPORT.pdf` | A document with a comprehensive overview, limitations, and long term guidelines. |
| `RESEARCH_NOTE.md` / `.pdf` | Short note: hypothesis, experiment, interpretation, recommendation. |
| `RESULTS.md` | Outcomes, configuration, model revision, seeds, caveats, and the commands that reproduce the runs. |
| `eval_results/b{2,5,10}_eval.md` | Raw holdout evaluation for each run. |
| `sv/README.md` | How the harness works and what each script does. |
| `sv/AGENT_PROMPT.md` | The task prompt the agent received. |
| `sv/runs/b{2,5,10}-test/` | Per-run config, rendered prompt, agent log, notes, forward-pass log, results, and the agent's scripts. |
| `upstream/` | Material from the original weight-space programming work that the agent could read. |

## Setup

```bash
uv sync                                    # one environment: torch, transformers, euroeval, langdetect
ln -s .venv .venv-eval                     # the scripts and docs call .venv-eval/bin/python; this points it at .venv
# Model, pinned revision (also recorded in RESULTS.md)
hf auth login                              # gemma is a gated model; need to accept license and authenticate
hf download google/gemma-3-270m-it --revision ac82b4e820549b854eebf28ce6dedaf9fdfa17b3 --local-dir gemma-3-270m-it
```

The experiment originally used a separate `.venv-eval` for `euroeval` and `langdetect`. They are now dependencies of
the main environment, and the symlink keeps the original command paths working. The data is not in the repository.
`make_data.py` creates it from the EuroEval Swedish datasets, which are private on the Hugging Face Hub and are read
through the `euroeval` package. See `sv/README.md` and `RESULTS.md` for the exact commands.

## Files not in the repository

Model weights are excluded. This covers `gemma-3-270m-it/`, every `edited/` directory, and the edit overlays
`sv/runs/*/edit/overlay.safetensors`. The overlays are backed up on Google Drive. To re-score a run, put its overlay at
`sv/runs/<run>/edit/overlay.safetensors`, then run `build_edited.py` and `final_eval.py` as described in `RESULTS.md`.
Evaluation data and caches are excluded as well.

## AI usage

No API based usage was made.

| Tool and model | Used for |
|---|---|
| Kiro CLI 3.5.0 with Claude Opus 5.5 | The editing agent in the three runs (`sv/run_agent.sh`). It analysed the model and wrote the edit scripts. |
| Kiro CLI assistant (Claude Sonnet 5.5 in the latest session; earlier sessions are not recorded) | Built the harness in `sv/`, ran the post-run analysis, and drafted `RESULTS.md`, `RESEARCH_NOTE.md` and this README. |
| Kiro CLI with Claude Haiku | One short test of the Kiro CLI flags before the runs. |

The experiment design, the choice of budgets, and the structure of the report were made by the author. The
harness, analysis and documents were written by the AI assistant under the author's direction.



## Model

The edited model is not pushed due to size restrictions and license concerns. Please contact the author for the adjusted weights to verify the numbers if needed.