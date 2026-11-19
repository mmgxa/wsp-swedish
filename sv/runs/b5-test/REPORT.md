# REPORT: run b5-test (budget 5 passes, tasks lm / scala / gen, Gemma-3-270M-it, Swedish)

## Read: hypothesis and evidence

**Pass 1 (generate, orig, 30 DEV gen + 34 hand-written Swedish prompts).** The original model answered 67% of DEV
questions in Swedish by the harness heuristic (en 10%, da/no 6.7%, other 16.7%) and 94% of my simple prompts. The
failures were language switches. Some answers started in Norwegian or Danish ("Det er vanskelig å si ...",
"Jeg er ikke sikker ...", "Indiens første kvinnelige ..."). Others came out in English when the question contained an
English song or film title. Repetition was 0%, so it was ruled out as the failure to target.

**Pass 2 (probe, MLP down_proj inputs, all 18 layers, all positions, orig).** The items were 16 hand-written parallel
paragraphs in Swedish, Norwegian, Danish and English, the 30 DEV lm texts, and the 64 pass-1 transcripts. All
further analysis used these arrays and the weights and cost no passes:
- Language-selective neurons are in layers 13-17, plus character-level ones in layer 0.
- Swedish neurons. L14 n1542 has mean activation +1.51 on Swedish, -0.14 on Norwegian, -0.12 on Danish and -0.09
  on English. L13 n72 has +1.36, -0.15, -0.13 and -0.01. Their output columns, read through the unembedding (with
  the post-FFN and final norm gains), favour Swedish-only tokens over Norwegian/Danish-only tokens and over
  English-only tokens. Each one carries 20-26% of its layer's MLP output norm on Swedish text and under 2% on
  English text. At the last prompt token they are weaker before answers that then drift. For L14 n1542: +1.45 before
  Swedish answers, +0.65 before the Norwegian one, -0.27 before the Danish one, +0.24 before the English ones.
- Norwegian/Danish neurons. These are the only late-layer neurons with activation ≥0.3 on Norwegian or Danish,
  |activation| ≤0.12 on Swedish and English, and an output that promotes Norwegian/Danish tokens. L15 n459 has
  +1.07 on Norwegian and ≤0.07 in absolute value elsewhere. L14 n1594 has +0.70 on Danish and ≤0.11 elsewhere. In the
  drift transcripts, L15 n459 fires on the token where the answer turns Norwegian ("Det" → " er": +0.4, then about
  +1.0 on the following tokens; "\n" → "Jeg": +0.6).

**Hypothesis.** The model's Swedish depends on a small set of late "language neurons". Its drift into
Norwegian/Danish comes from Norwegian/Danish neurons outweighing weak Swedish ones. Two edits follow from this:
amplify the Swedish neurons, or silence the Norwegian/Danish ones.

**Pass 3 (score, both): Swedish neurons ×2. Refuted.** DEV lm bpc went 1.957 → 2.010 (+0.053, 8/30 texts better).
Swedish paragraphs got +0.055, Norwegian +0.036, Danish +0.033 and English +0.002. scala accuracy went 0.633 → 0.567
(two right→wrong) while choice_nll went 0.891 → 0.827. The neurons are Swedish-specific, since English did not move,
but pushing them harder makes Swedish text less likely. I dropped this edit.

**Pass 4 (score, both): Norwegian/Danish neurons removed. Mechanism confirmed, Swedish gain tiny.** Norwegian bpc
rose +0.029 and Danish +0.037, so the neurons do carry Norwegian/Danish. Swedish moved −0.003 on DEV lm (21/30 texts
better) and −0.0035 on my Swedish paragraphs. English rose +0.0026 (0.26%). scala accuracy stayed at 0.633 with no
flips, and choice_nll rose +0.013.

**Pass 5 (generate, both, same 64 prompts as pass 1).** DEV sv_rate went 0.667 → 0.733 and da/no 0.067 → 0.033. Only
one switch is real: the Norwegian "Det er vanskelig å si ..." became the Swedish "Det är en fråga som ...". The second
en→sv change is a labelling artifact: the edited answer is a repeated Danish-looking line, and it is also the one new
repetition (repeat_rate 0 → 0.033). One Danish answer stayed Danish. On my own prompts sv_rate stayed at 0.94. 41 of 64
answers were identical, and the rest were small rewordings.

## Write: the submitted edit (edit B, `work/edit.py`)

The edit zeroes the `down_proj` output column of two MLP neurons: layer 14 neuron 1594 (Danish) and layer 15 neuron
459 (Norwegian). In plain words, these two neurons can no longer write anything into the residual stream. They are
nearly silent on Swedish and English text, so the edit only acts where the model is producing Norwegian/Danish-like
continuations. The edit touches 2 tensors and changes 1,280 weights after bf16 rounding (0.00048% of 268M).
Embeddings, the final norm and the architecture are untouched. The rejected edit A, used only in pass 3, multiplied the
`down_proj` columns of L13 n72 and L14 n1542 by 2.

## Results (HOLDOUT, `sv/final_eval.py`, run once on edit B)

```
b5-test: 5/5 passes; 0.0005% of weights changed (2 tensors)
task          n  metric           orig     edit    delta  95% CI of delta
lm           49  bits/char       1.896    1.895   -0.001  [-0.002, +0.000]
scala        50  acc             0.560    0.560   +0.000  [+0.000, +0.000]
                 choice_nll      1.072    1.106   +0.034  [+0.013, +0.057]
gen          50  sv_rate         0.580    0.580   +0.000  [+0.000, +0.000]
gen          50  repeat_rate     0.040    0.040   +0.000  [+0.000, +0.000]
kl_generic_prompts_nats: 0.0012
langdetect (independent): {'orig': {'sv': 0.58, 'en': 0.1, 'da': 0.0, 'no': 0.06, 'other': 0.26}, 'edit': {'sv': 0.58, 'en': 0.08, 'da': 0.0, 'no': 0.06, 'other': 0.28}}
```

| Target | Outcome |
|---|---|
| lm bpc lower, CI excludes 0 | −0.001, CI [−0.002, +0.000] touches 0: **not met** |
| gen sv_rate higher, langdetect agrees | 0.58 → 0.58; langdetect sv 0.58 → 0.58: **not met** |
| mean mc acc not lower | scala acc 0.56 → 0.56 (met); scala choice_nll worse, +0.034 [+0.013, +0.057] |
| lm_en ≤ 2% worse, KL ≤ 0.1 | lm_en is not in this run's tasks (DEV English paragraphs: +0.26%); KL 0.0012 nats: met |
| weights changed ≤ 1% | 0.00048%: met |
| passes ≤ 5 | 5: met |

## Forward-pass summary

| Pass | Kind | Which | Items | Edit | Question → answer |
|---|---|---|---|---|---|
| 1 | generate | orig | 30 gen + 34 own | none | How does Swedish fail? → language drift (no/da/en), no repetition |
| 2 | probe mlp | orig | 158 texts | none | Which neurons are language-selective? → Swedish L13 n72, L14 n1542; Norwegian L15 n459, Danish L14 n1594 |
| 3 | score | both | 30 lm, 30 scala, 64 paragraphs | A: Swedish neurons ×2 | Does amplifying Swedish help? → no, bpc +0.053 |
| 4 | score | both | same | B: no/da neurons ×0 | Is removal free for Swedish/English? → yes; no/da bpc +0.03 |
| 5 | generate | both | same as pass 1 | B | Does drift become Swedish? → 1 real switch of 30 DEV items |

## Limitations

- **No measurable Swedish gain on the HOLDOUT.** The edit B mechanism is real: removing the two neurons makes
  Norwegian/Danish text 2-2.5% less likely and barely touches English. But the drifts in the model's answers are mostly
  not switched by these two neurons. The DEV generation gain (one real flip in 30) did not transfer to the HOLDOUT
  (0.58 → 0.58). The Swedish bpc change is within noise on both splits.
- **scala choice_nll got worse** on the HOLDOUT (+0.034, CI excludes 0), with the same accuracy. On DEV it was +0.013.
  This is a small but real cost on the ja/nej acceptability prompts. I have no mechanism for it.
- **The edit is not a general change.** It is language-specific by construction: KL on generic prompts is 0.0012 nats
  and the DEV English bpc change is +0.26%. Its intended effect is the Norwegian/Danish-to-Swedish switch, and that
  effect turned out too weak to see on 50 holdout prompts.
- **Budget limits.** With 5 passes I tested only two edits, each at one strength. I could not run a dose-response, for
  example Swedish neurons at ×1.2, or zeroing the wider set of neurons with Norwegian/Danish-signed outputs (L15 n1944,
  L16 n570, whose negative activations push Norwegian/Danish). I also did not localize the English-title failure. The
  amplification result (×2 is worse) says the Swedish neurons are not under-driven at this strength. It does not
  exclude a smaller gain at a weaker strength.
- **Probe texts.** I wrote the Norwegian/Danish/English parallel paragraphs myself (16 per language, about 650 tokens
  each). The neuron selection rests on this small sample plus the DEV lm texts and the model's own transcripts.
- **DEV versus HOLDOUT.** DEV: lm −0.003 bpc, gen sv_rate +0.067 (1 real flip plus 1 artifact). HOLDOUT: lm −0.001,
  gen sv_rate +0.000. The DEV gen gain was mostly noise.
- **Audit failure not caused by this agent.** `verify.py` reports "final_eval ran 2 time(s) on 2 distinct edit(s)".
  The first line of `final_eval_log.jsonl` is dated 2026-11-19T06:26:29 with `overlay_hash: "none"`. That is before
  this run had any edit: the first overlay was written about 06:27, and pass 3 ran after it. I ran `final_eval.py`
  exactly once, at 06:30:15, on edit B (`1f13980da1ef24c5`). The extra entry is most likely another process running
  `sv/final_eval.py` without `WSP_RUN_DIR`. `sv/runs/CURRENT` points to b5-test, and the b2-test run shows its own
  final_eval at 06:27:36, one minute later. That entry evaluated the unedited model against itself, so it carries no
  information for choosing an edit, and none was chosen from it. I left the log untouched rather than edit an audit
  file.
- The harness cache `cache/` holds the full pass-2 probe arrays (about 1.4 GB). I left it in place because the
  harness owns it.

## verify.py output

```
run b5-test
  ok    harness files unchanged
  ok    no forbidden API in 16 agent script(s)
  ok    5/5 passes used
        pass 1: generate orig 64 items {'gen': 30, 'gen_own': 34} edit=none 7.5s pass1: diagnose orig generations
        pass 2: probe orig 158 items {'?': 158} edit=none 8.5s pass2: mlp neuron language selectivity
        pass 3: score both 124 items {'lm': 30, 'scala': 30, 'par_sv': 16, 'par_no': 16, 'par_da': 16, 'par_en': 16} edit=46563ee52cd2f5c4 6.6s pass3: S=2.0 on L13n72,L14n1542
        pass 4: score both 124 items {'lm': 30, 'scala': 30, 'par_sv': 16, 'par_no': 16, 'par_da': 16, 'par_en': 16} edit=1f13980da1ef24c5 6.4s pass4: zero L14n1594,L15n459
        pass 5: generate both 64 items {'gen': 30, 'gen_own': 34} edit=1f13980da1ef24c5 8.4s pass5: gen both, zero L14n1594,L15n459
  ok    edit: 2 tensors, 1280 weights (0.0005%) change after bf16 rounding
  ok    work/edit.py reproduces the overlay byte for byte
  ok    embed_tokens and final norm identical in edited/
  FAIL  final_eval ran 2 time(s) on 2 distinct edit(s): the holdout was used to choose between edits


b5-test: 5/5 passes; 0.0005% of weights changed (2 tensors)
task          n  metric           orig     edit    delta  95% CI of delta
lm           49  bits/char       1.896    1.895   -0.001  [-0.002, +0.000]
scala        50  acc             0.560    0.560   +0.000  [+0.000, +0.000]
                 choice_nll      1.072    1.106   +0.034  [+0.013, +0.057]
gen          50  sv_rate         0.580    0.580   +0.000  [+0.000, +0.000]
gen          50  repeat_rate     0.040    0.040   +0.000  [+0.000, +0.000]
kl_generic_prompts_nats: 0.0012
langdetect (independent): {'orig': {'sv': 0.58, 'en': 0.1, 'da': 0.0, 'no': 0.06, 'other': 0.26}, 'edit': {'sv': 0.58, 'en': 0.08, 'da': 0.0, 'no': 0.06, 'other': 0.28}}

1 failure(s), 0 warning(s)
```
(The FAIL refers to the 06:26:29 `overlay_hash: "none"` entry explained under Limitations.)
