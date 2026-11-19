# upstream/

Files copied unmodified from https://github.com/RikiyaT/lm_understand-logs (the authors' logs for the blog post
"Sparks of Weight-Space Programming"). Data is CC BY 4.0, scripts are MIT (see `LICENSE`). Attribution: Takehi, Teoh,
Zhang, Khattab.

| File | Origin | Use here |
|---|---|---|
| `exp7_sycophancy/dataset/facts.json` | `exp7_sycophancy/dataset/facts.json` | 452 simple facts (12 types) that the base Gemma-3-270M-it answers correctly. The task's data source. |
| `exp7_sycophancy/dataset/data.py` | same | How the facts were built and how wrong guesses (`swap`) are assigned: another fact's answer of the same type. |
| `exp7_sycophancy/code/eval.py` | `exp7_sycophancy/code/eval.py` | Reference log-probability sycophancy evaluation. |
| `exp8_math/gemma_math_first_message.txt` | `exp8_math/gemma_followup/runs/g10b/turns/t001/prompt.txt` | The exact first message the authors sent for the Gemma 270M weight-editing run. The structure of `PROMPT.md` follows it. |
