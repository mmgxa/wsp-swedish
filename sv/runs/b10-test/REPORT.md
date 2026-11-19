# Weight-space programming for Swedish, run b10-test (budget 10 passes, 9 used)

Summary: I found no switchable Swedish failure in Gemma-3-270M-it. The submitted edit removes three MLP neurons that
fire only on Scandinavian text. On DEV it lowered Swedish bits per character by 0.2-0.3% and left English unchanged. On
HOLDOUT the bpc change is -0.001, with a 95% interval that includes zero. The largest effect I measured is general:
the model is over-confident on raw text in every language. Fixing that would be a global change to the output
distribution, which is out of scope, so I did not edit it.

## Read: hypotheses and evidence

The sets were fixed before any edit. DEV is the harness DEV split (30 items each of lm, scala, gen). SELF is a set of
hand-written items in `work/items.py`: 30 Swedish and 6 English prompts, 16 parallel sv/en/da/no sentences and 12
English texts. I chose neurons on Swedish parallel sentences plus DEV lm items 0-14 (A), and used items 15-29 (B) as
an internal check. HOLDOUT was used only through `final_eval.py`, once.

| Hypothesis | Evidence | Passes | Verdict |
|---|---|---|---|
| Swedish prompts are answered in English, Danish or Norwegian | DEV gen: 22/30 sv, 1 en (an echoed English song title), 1 no, 6 "other" (short answers). SELF: 28/30 sv | 1 | Rare. Little room to gain |
| Repetition and degeneration | 3/30 DEV answers loop. Many answers are fluent but nonsensical Swedish (knowledge limit) | 1 | Present but small. Not targeted |
| Language confusion lowers Swedish likelihood | At Swedish function-word positions the original model gives the Danish/Norwegian form 0.29% and the English form 0.02% of the probability (correct word: 22%). Top-1 errors are other Swedish words | 3 (free analysis of its output) | Refuted |
| A component hurts Swedish text likelihood | L17 n1977 looked harmful by a direct-logit test against the mean embedding. Removing it (E1) worsened every language by 5-9% bpc | 2, 3 | Refuted for n1977. The test was wrong |
| (found) Over-confidence on raw text | NLL would fall by 9% (Swedish, 4.85 -> 4.41 nats/token) and by 10% (English, 3.17 -> 2.85) if the logits were scaled by 0.7. The best scale is the same, 0.70, for both languages | 3 (free analysis) | Real, general, not Swedish-specific. Out of scope |
| A few Scandinavian-only neurons lower the correct Swedish token's probability, apart from any temperature effect | E2 (45 neurons, first-order rule) gained only by flattening the distribution: DEV A -1.4%, B +0.4%, KL 0.2-0.26. A temperature-free first-order rule picked 3 neurons (E3): DEV A -0.07%, B -0.30%, entropy unchanged, English KL 0.0004. Larger sets made things worse (E4 +0.8%, E5 +0.02/+0.16%) | 2, 4-7, 9 | Weak support on DEV. Not significant on HOLDOUT |

Other observation (harness interaction, not edited): almost every batched greedy answer starts with the glitch token
"ovce" (id 222101). It appears when the prompt is left-padded in the batch (62 of 63 padded prompts) and never for the
3 unpadded prompts. It affects the original and the edited model alike, on DEV and on HOLDOUT. 2/30 DEV answers
consist only of this token and are counted as "other". The cause seems to lie in the padded generation path, not in
Swedish. Model weights cannot reach the pad embedding.

How the neuron criterion evolved:
1. Pass 2 recorded all MLP activations on Swedish, English, Danish and Norwegian text. The first ranking (logit of
   the correct token minus the vocabulary-mean logit) pointed at L17 n1977. That neuron has a near-constant activation
   and carries 22% of the layer-17 MLP output. E1 showed the ranking was wrong.
2. `an4.py` measures the first-order change of log p(correct token) when a neuron is removed. It uses the model's own
   next-token distribution and both RMSNorm couplings (post_feedforward_layernorm and model.norm). With these it
   predicts the sign of E1 in all 7 text groups.
3. E2 showed that this rule mostly picks neurons whose removal flattens the distribution. `an9.py` splits the
   first-order gain into a temperature part and a temperature-free part. That split predicted E2's DEV B result.
4. E3 has positive temperature-free gain on all three Swedish groups, no English cost, and fires only on
   Scandinavian text (|activation| at least 3x higher than on English). E4 relaxed the rule and included neurons
   whose removal sharpens the distribution, which hurt. E5 (both parts positive) added only L17 n1364 and was worse
   than E3.

## Write: the edit (E3, `work/edit.py`)

In `model.layers.15.mlp.down_proj.weight` column 459 and `model.layers.17.mlp.down_proj.weight` columns 736 and 1067,
all values are set to 0. This removes three MLP neurons. Each one is active on Swedish, Danish and Norwegian text and
nearly silent on English (mean |activation| on Swedish 0.08-0.23, on English 0.02-0.03). L17 n1067 reads most easily:
when its activation is positive it suppresses the Swedish determiners den/det/de/Den/Det, and when negative it
promotes them. Its first-order effect on the correct Swedish token is negative on average. The edit changes 1,920
weights (0.0007% of 268M) in 2 tensors.

## Results (HOLDOUT, `sv/final_eval.py`, run once on the frozen E3)

```
b10-test: 9/10 passes; 0.0007% of weights changed (2 tensors)
task          n  metric           orig     edit    delta  95% CI of delta
lm           49  bits/char       1.896    1.895   -0.001  [-0.003, +0.002]
scala        50  acc             0.560    0.540   -0.020  [-0.060, +0.000]
                 choice_nll      1.072    1.027   -0.045  [-0.078, -0.015]
gen          50  sv_rate         0.580    0.580   +0.000  [-0.060, +0.060]
gen          50  repeat_rate     0.040    0.100   +0.060  [-0.020, +0.160]
kl_generic_prompts_nats: 0.0146
langdetect (independent): {'orig': {'sv': 0.58, 'en': 0.1, 'da': 0.0, 'no': 0.06, 'other': 0.26}, 'edit': {'sv': 0.6, 'en': 0.1, 'da': 0.0, 'no': 0.06, 'other': 0.24}}
```

Against the success criteria:

| Criterion | Result |
|---|---|
| lm bpc lower, interval excludes 0 | -0.001, interval [-0.003, +0.002] includes 0. **Not met** |
| gen sv_rate higher, langdetect agrees | Heuristic sv_rate unchanged (0.58). langdetect 0.58 -> 0.60 (one answer). **Not met** |
| mean mc acc not lower | scala is the only mc task: acc 0.56 -> 0.54 (one item, interval [-0.06, 0]). **Not met** (choice NLL improved, interval excludes 0) |
| lm_en at most 2% worse, KL at most 0.1 | lm_en is not in this run's HOLDOUT tasks. On DEV, SELF English bpc changed by -0.13% (harness scorer). KL on generic prompts 0.015. **Met** |
| weights changed at most 1% | 0.0007%. **Met** |
| passes at most 10 | 9. **Met** |

## Forward-pass summary

| Pass | Kind | Edit | Question | Answer |
|---|---|---|---|---|
| 1 | generate orig, 66 | none | Wrong language, repetition? | Mostly Swedish. 1 no, 1 en, 3/30 loops. "ovce" padding glitch |
| 2 | probe mlp orig, 106 texts | none | Which neurons act on Swedish vs other languages? | Activations for all 36,864 neurons, used for every ranking |
| 3 | probe hidden both, 232 | E1: L17 n1977 = 0 | Does the "English-bias" neuron limit Swedish? | No: +5..9% bpc in all languages. This pass also gave the full orig distributions |
| 4 | probe hidden both, 232 | E2: 45 neurons = 0 | Do first-order harmful Scandinavian neurons limit Swedish? | DEV A -1.4%, B +0.4%. The gain is flattening only. Rejected |
| 5 | probe hidden both, 232 | E3: 3 neurons = 0 | Same, temperature-free rule | DEV A -0.07%, B -0.30%, sv parallel -1.2%, English unchanged |
| 6 | probe hidden both, 232 | E4: 14 neurons = 0 | Larger set? | +0.8% on both halves (sharpening). Rejected |
| 7 | probe hidden both, 232 | E5: E3 + L17 n1364 | Add the one more neuron that passes both tests? | Worse than E3. Rejected. E3 frozen |
| 8 | generate both, 66 | E3 | Does generation change? | DEV sv 0.733 -> 0.667 (2 answers), repeat 0.10 -> 0.067. Neutral within noise |
| 9 | score both, 88 | E3 | Does the harness scorer agree? | DEV lm bpc 1.9580 -> 1.9537, 23/30 texts better. English -0.13%. scala acc -1 item |

The probe passes return final hidden states after model.norm. I computed logits from them with the tied embedding
(free analysis of fwd output). Pass 9 used the harness scorer and agreed with these numbers.

## Limitations

- **The DEV gain did not replicate on HOLDOUT at a detectable size.** DEV lm: -0.22% (harness scorer), 23/30 texts
  better. HOLDOUT lm: -0.05%, interval includes 0. The edit was selected on DEV text (including DEV lm B in the final
  rule), so part of the DEV gain is selection.
- The effect is Swedish-specific by construction, because the three neurons are almost silent on English, and the
  English KL is 0.0004 per token. It is not Swedish-only, though: it also helps Danish (-0.9%) and hurts Norwegian
  (+2.7% on the parallel sentences).
- scala: choice NLL improved on DEV and on HOLDOUT (interval excludes 0), but accuracy dropped by one item on each.
  The better choice NLL may come from the edit making the model less confident in the wrong answer.
- Generation: greedy answers change wording often (43/66 on DEV differ somewhere in 48 tokens) because greedy
  decoding diverges at near-ties. HOLDOUT repeat_rate rose from 2/50 to 5/50 (interval includes 0). I cannot rule out
  that the edit encourages loops slightly.
- The one large, robust lever I found is general over-confidence on raw text: the best logit scale is 0.7 for both
  Swedish and English. I did not edit it because it is not a Swedish obstacle and fixing it would be a global
  temperature change. A neuron-level version of it (for example amplifying a constant "entropy" neuron such as L17
  n1977, whose removal sharpened every language) is a possible follow-up, but it is out of scope here.
- The first-order criteria use only the direct path of each neuron, from MLP activations recorded in one pass. They
  overestimate the size of effects and have the right sign only for small edits.
- The batched generation's "ovce" glitch caps gen sv_rate for both models. It is a harness/padding interaction and
  I left it alone.
- Small sets (30 DEV and about 50 HOLDOUT items per task) cannot resolve effects below about 0.5% bpc or about 10
  points of rate.
- Files: the raw activation and hidden-state dumps were deleted to keep the run small. The harness cache in
  `cache/` still holds every pass result, and rerunning the pass scripts with the same candidate (`work/edit.py
  <name>`) serves them from that cache for free.

## verify.py

```
run b10-test
  ok    harness files unchanged
  ok    no forbidden API in 27 agent script(s)
  ok    9/10 passes used
        pass 1: generate orig 66 items {'gen': 30, 'gen_sv_self': 30, 'gen_en_self': 6} edit=none 8.1s P1 baseline generations: language and repetition
        pass 2: probe orig 106 items {'?': 106} edit=none 8.9s P2 mlp acts sv/en/da/no + DEV lm
        pass 3: probe both 232 items {'?': 232} edit=505f6ffd4e58ae55 15.6s p3_E1: final hidden both, bpc/scala/KL
        pass 4: probe both 232 items {'?': 232} edit=bfea6063185cb83c 15.4s p4_E2: final hidden both, bpc/scala/KL
        pass 5: probe both 232 items {'?': 232} edit=b2de7462228fdec0 15.3s p5_E3: final hidden both, bpc/scala/KL
        pass 6: probe both 232 items {'?': 232} edit=16130648646be83f 15.2s p6_E4: final hidden both, bpc/scala/KL
        pass 7: probe both 232 items {'?': 232} edit=200c93106238c9ee 15.1s p7_E5: final hidden both, bpc/scala/KL
        pass 8: generate both 66 items {'gen': 30, 'gen_sv_self': 30, 'gen_en_self': 6} edit=b2de7462228fdec0 11.3s P8 E3 generations both
        pass 9: score both 88 items {'lm': 30, 'scala': 30, 'lm_en_self': 12, 'lm_par_sv': 16} edit=b2de7462228fdec0 6.4s P9 E3 official scorer both
  ok    edit: 2 tensors, 1920 weights (0.0007%) change after bf16 rounding
  ok    work/edit.py reproduces the overlay byte for byte
  ok    embed_tokens and final norm identical in edited/
  ok    final_eval ran 1 time(s) on 1 distinct edit(s)

b10-test: 9/10 passes; 0.0007% of weights changed (2 tensors)
task          n  metric           orig     edit    delta  95% CI of delta
lm           49  bits/char       1.896    1.895   -0.001  [-0.003, +0.002]
scala        50  acc             0.560    0.540   -0.020  [-0.060, +0.000]
                 choice_nll      1.072    1.027   -0.045  [-0.078, -0.015]
gen          50  sv_rate         0.580    0.580   +0.000  [-0.060, +0.060]
gen          50  repeat_rate     0.040    0.100   +0.060  [-0.020, +0.160]
kl_generic_prompts_nats: 0.0146
langdetect (independent): {'orig': {'sv': 0.58, 'en': 0.1, 'da': 0.0, 'no': 0.06, 'other': 0.26}, 'edit': {'sv': 0.6, 'en': 0.1, 'da': 0.0, 'no': 0.06, 'other': 0.24}}

0 failure(s), 0 warning(s)
```
