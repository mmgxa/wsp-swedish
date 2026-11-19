# Notes (write the exact current edit and what changed BEFORE every pass)

## Pass 1 (planned): generate, which=orig
- Current edit: none (no overlay; original weights).
- Items: 30 DEV `gen` items + 34 hand-written Swedish open-ended prompts (task `gen_own`, in work/common.py).
- Question: how does the ORIGINAL model fail on Swedish prompts? Rate of English / Danish / Norwegian answers,
  repetition, empty answers. Is there a switchable failure (e.g. language switch to English) worth targeting?
- Result: DEV gen sv_rate 0.667, en 0.10, da/no 0.067, other 0.167 (mostly short answers or Norwegian-looking text
  labelled other), repeat 0. Own prompts sv 0.94, en 0, other 0.06. Failures seen: Norwegian/Danish drift
  ("Det er vanskelig å si", "Jeg er ikke sikker", "Indiens første kvinnelige", "Hver höst") and English answers when
  the prompt contains an English song/film title. No repetition problem at 48 tokens.

## Pass 2 (planned): probe mlp (down_proj inputs), all 18 layers, positions=all, which=orig
- Current edit: none (original weights).
- Items: 16 hand-written parallel paragraphs x 4 languages (sv, no, da, en; work/texts.py), the 30 DEV `lm` Swedish
  texts, and the 64 prompt+answer transcripts from pass 1 (chat template + the original model's greedy answer).
- Question: which MLP neurons are language-selective (fire on Norwegian/Danish or English but not on Swedish text,
  and vice versa)? Do the Norwegian/Danish-selective neurons also fire inside the model's own Norwegian-drift answers
  and inside its Swedish answers? This localizes candidate components for an edit; their output weights are then
  read through the unembedding (free) to check what tokens they promote.
- Result (analysis free, from the pass-2 arrays): two late MLP neurons are Swedish-specific. L14 n1542: mean act
  +1.51 on Swedish paragraphs, +1.29 on DEV lm text, -0.14 Norwegian, -0.12 Danish, -0.09 English. L13 n72: +1.36 /
  +1.23 / -0.15 / -0.13 / -0.01. Their down_proj columns, read through the unembedding, favour Swedish-only tokens
  over Norwegian/Danish-only tokens (lens -45.9, -39.4) and over English-only tokens (-49.2, -37.5). Each carries
  ~20-26% of its layer's MLP output norm on Swedish text, ~0-2% on English text. At the last prompt token (before the
  answer starts) they are lower when the original model then drifts: L14 n1542 = +1.45 (52 Swedish answers),
  +0.65 (Norwegian), -0.27 (Danish), +0.24 (3 English answers); L13 n72 = +0.75 / +0.01 / -0.18 / +0.06.
  A Norwegian neuron L15 n459 (+1.08 on Norwegian, ~0 elsewhere) fires in both Norwegian/Danish drift answers.

## Pass 3 (planned): score, which=both
- Current edit (work/edit.py, S=2.0): model.layers.13.mlp.down_proj.weight column 72 x 2.0 and
  model.layers.14.mlp.down_proj.weight column 1542 x 2.0. 1280 weights (0.00048%) change after bf16 rounding.
- Items: 30 DEV lm, 30 DEV scala, plus my 16 parallel paragraphs as lm items per language (tasks par_sv, par_no,
  par_da, par_en).
- Question: does amplifying the Swedish neurons lower Swedish bits/char (lm, par_sv) without raising English bpc
  (par_en)? Mechanism check: Norwegian/Danish bpc should change little (neurons are off there) or rise. scala
  choice_nll should not get worse.
- Result: x2 on the Swedish neurons made Swedish WORSE: DEV lm bpc 1.957 -> 2.010 (+0.053, only 8/30 items better),
  par_sv +0.055, par_no +0.036, par_da +0.033, par_en +0.002; scala acc 0.633 -> 0.567 (2 right->wrong, 0
  wrong->right) although choice_nll fell 0.891 -> 0.827. Amplification is rejected (edit A abandoned; it was
  `W[:, idx] *= 2.0` on L13 col 72 and L14 col 1542).

## Pass 4 (planned): score, which=both, same items as pass 3
- Current edit (work/edit.py, edit B): model.layers.14.mlp.down_proj.weight column 1594 x 0 and
  model.layers.15.mlp.down_proj.weight column 459 x 0 (two no/da-selective neurons removed). 1280 weights (0.00048%).
  Edit A is fully reverted (edit.py starts from the original weights).
- Question: is removing the Norwegian/Danish neurons free for Swedish and English likelihood (lm, par_sv, par_en, scala
  roughly unchanged), and does it raise par_no / par_da bpc (mechanism check that they carry no/da)?
- Result: DEV lm bpc 1.9572 -> 1.9543 (-0.0029, 21/30 items better), par_sv -0.0035 (11/16 better), par_no +0.029,
  par_da +0.037 (mechanism confirmed: removing them hurts exactly Norwegian/Danish), par_en +0.0026 (0.26%),
  scala acc 0.633 -> 0.633 (no flips), choice_nll +0.013.

## Pass 5 (planned, last): generate, which=both
- Current edit: edit B unchanged (L14 down_proj col 1594 x 0, L15 down_proj col 459 x 0; 1280 weights).
- Items: same 64 as pass 1 (30 DEV gen + 34 own prompts).
- Question: does removing the no/da neurons turn the Norwegian/Danish drift answers into Swedish (sv_rate up,
  da_no_rate down) without new English answers, repetition, or broken text in answers that were already Swedish?
- After this pass the edit is frozen as is, whatever the result (no passes left to test anything else).
- Result: DEV gen sv_rate 0.667 -> 0.733, da_no 0.067 -> 0.033, en 0.100 -> 0.067, repeat 0 -> 0.033. Real flip:
  gen-15 Norwegian -> Swedish ("Det er vanskelig å si ..." -> "Det är en fråga som ..."). gen-27 en -> sv is a
  label artifact: the edited answer is a repeated Danish-looking line ("Hjertet, hjertet, så let"), which is also the
  new repetition. gen-24 stays Danish. Own prompts: sv 0.941 both, 41/64 answers identical, the rest small rewordings.

## Frozen: edit B (L14 down_proj col 1594 x 0, L15 down_proj col 459 x 0). Next: sv/final_eval.py once.
