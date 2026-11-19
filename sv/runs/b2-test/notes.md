# Notes (write the exact current edit and what changed BEFORE every pass)

## Before pass 1 (2026-11-19)

- Current edit: none (`edit_hash=none`, no overlay). Pass is `which="orig"`, so the edit does not matter.
- Free analysis done so far (`work/weights_analysis.py`, no pass): a Swedish-minus-English output direction `d` from the
  tied embedding (154 Swedish vs 164 English single-token function words). No layer's `o_proj` or `down_proj` writes
  along `d` more than a random direction would (ratios 0.014-0.035 vs 1/sqrt(640) = 0.040). Weights alone show no
  "anti-Swedish" component, so I need activations.
- Pass 1: `fwd("probe", what="mlp", layers=0..17, positions="all", which="orig")` on raw texts:
  70 Swedish texts = the `sv/data/default/dev/lm.jsonl` (EuroEval train split) items that are NOT in the tiny DEV `lm`
  set, plus 50 English texts = first 50 of `sv/data/default/dev/lm_en.jsonl`. The 30 tiny DEV `lm` items are kept out
  so pass 2 can test the edit on texts that did not choose it.
- Question: which MLP neurons, through their direct effect on the logit of the actual next token (direct logit
  attribution, DLA), push down the correct Swedish tokens consistently across texts, without helping English? If such
  neurons exist and replicate between two halves of the Swedish texts, damping them is the edit.

## Before pass 2

- Current edit (`work/edit.py`, edit_hash `473d5ecd84a756c1`): zero the `down_proj` column (the output vector) of 19
  MLP neurons, strength 0 (full removal):
  L12 n855; L13 n1771; L14 n973, n1494; L16 n160, n598, n754, n999, n1403, n1573;
  L17 n170, n216, n244, n458, n666, n798, n888, n1143, n1188.
  5 tensors, 12,160 weights changed after bf16 rounding (0.0045% of 268M). Embeddings, lm_head, final norm untouched.
- What changed since pass 1: everything (pass 1 had no edit). Neurons chosen by `work/select_neurons.py` from pass-1
  data only (criterion in that file).
- Pass 2: `fwd("score", which="both")`, one pass, about 770 items:
  - tiny DEV `lm` (30 Swedish texts, NOT used for selection): the clean test of the hypothesis;
  - the 70 Swedish selection texts as `lm` (in-sample check that direct attribution predicts the real total effect);
  - `lm_en`: the 50 English texts of `default/dev/lm_en.jsonl` NOT used in pass 1 (retention);
  - `scala`: tiny DEV 30 + the other 70 of `default/dev/scala.jsonl`;
  - retention mc tasks from `default/dev`: swerec, hellaswag, arc, mmlu, belebele (100 each).
- Question: does removing these neurons lower Swedish bits per character on texts it was not chosen on, without
  raising English bpc by more than 2% or lowering mc choice_nll / accuracy?
- Decision rule fixed now: there is no pass left to tune, so whatever pass 2 shows, this exact edit is the frozen
  candidate for `final_eval.py`. The result only decides what I claim in the report.

## After pass 2 (budget spent, 2/2)

- Correction to the plan above: `default/dev/scala.jsonl` shares only 6 of its 100 ids with tiny DEV scala, so
  `scala` had 124 items (30 + 94), not 100. Total 774 items.
- An OOM warning from the CUDA allocator appeared during the pass; the pass finished and all items have results.
- Result (paired, same items): Swedish bpc worse. tiny-DEV `lm` 1.9585 -> 2.0493 (+0.091, 95% CI [+0.079, +0.105],
  29/30 texts worse); selection texts +0.079; English `lm_en` +0.019 (+1.5%). scala choice_nll +0.073, acc unchanged.
  The four general mc tasks: choice_nll about -0.5 nats better, accuracy unchanged within 1 item.
- Frozen candidate, per the rule fixed before pass 2: the same edit, `edit_hash 473d5ecd84a756c1`.
