# Notes (write the exact current edit and what changed BEFORE every pass)

## Sets (fixed before any edit)
- DEV: the harness DEV items, 30 each of lm, scala, gen (EuroEval train split). All edit decisions use these.
- SELF: my own hand-written items in `work/items.py` (Swedish instruction prompts, Swedish/English/Danish/Norwegian
  parallel sentences, a few English texts for an English retention check). Used for diagnosis and as a second check.
- HOLDOUT: only through `sv/final_eval.py`, once, after freezing.

## Pass 1 (planned)
- Current edit: none (no overlay).
- Request: generate, which="orig", 30 DEV gen prompts + 30 SELF Swedish prompts + 6 SELF English prompts.
- Question: when the original model gets a Swedish prompt, does it answer in English / Danish / Norwegian, does it
  repeat or degenerate, or does it answer in Swedish? Which failure (if any) is common enough to be worth repairing?

## Pass 1 result (generate, orig)
DEV gen: sv 0.733, en 0.033 (1), da/no 0.033 (1 Norwegian), other 0.20, repeat 0.10. SELF sv: sv 0.93.
Almost every answer starts with the glitch token "ovce" (id 222101). It appears exactly when the prompt is
left-padded in the batched generate (pad=0 -> no "ovce" in 3/3 cases; pad>=1 -> "ovce" in 62/63). 3 answers are only
"ovce" (empty answers). The other "other" labels are short answers without function words.

## Pass 2 (planned)
- Current edit: none.
- Request: probe, what="mlp", layers 0..17, positions="all", which="orig". Items (raw text, no chat template):
  16 parallel sentences x sv/en/da/no (64), DEV lm items 0-29 (30), SELF English texts (12). 106 items.
- Question: which MLP neurons are active on Swedish text and push the next-token logits away from Swedish
  (towards Danish/Norwegian or English function words, or against the observed next token)? Neuron selection will
  use parallel sentences + DEV lm items 0-14 only; DEV lm items 15-29 stay as an internal check.

## Pass 2 result (probe mlp, orig)
Direct effect (neuron activation x its output direction through post_feedforward_layernorm and final norm, on the
observed next token, relative to the mean embedding) ranked over Swedish parallel sentences + DEV lm 0-14:
L17 n1977 is the strongest neuron that hurts Swedish next tokens but helps English: -522 (sv), -463 (DEV lm 15-29,
held out of the selection), -489 (da/no), +90 (en). Its activation is almost constant (-1.9 +- 0.6 on Swedish,
-1.8 +- 0.3 on English), so it acts as a fixed bias; with the negative activation it promotes English content words.
It carries ~22% of the layer-17 MLP output norm. Next strongest Swedish-harmful, English-neutral neurons are 4-7x weaker
(L16 n940, n973, n52; L17 n1364, n798).

## Pass 3 (planned)
- Current edit E1: model.layers.17.mlp.down_proj.weight column 1977 set to 0 (neuron L17 n1977 removed). 640 weights.
- Request: probe, what="hidden", layers=[17] (final, after model.norm), positions="all", which="both". Items (232):
  DEV lm 30, SELF English 12, parallel sv/en/da/no 64, DEV scala 30 x 2 choices as full chat texts, DEV gen prompts 30,
  SELF sv prompts 30, SELF en prompts 6. From the final hidden states I compute (with the tied embedding) bpc per group,
  scala choice NLL/acc, KL(orig||edit) per token and at the first answer token of each prompt.
- Question: does removing L17 n1977 lower Swedish bpc (DEV lm A and B, parallel sv), and what does it cost on English
  text and in KL at the first answer token?

## Pass 3 result (probe hidden both, E1)
E1 (remove L17 n1977) is refuted: bpc +7.5% DEV lm A, +9.0% DEV lm B, +8.0% SELF English, +4.9..8.4% parallel
sv/en/da/no; scala choice NLL 0.910 -> 0.881 (acc same 0.633); KL per token 0.18 (Swedish), first-answer-token KL 0.06.
The neuron is a constant bias that helps all languages; my pass-2 criterion (logit vs the mean embedding) was wrong.
New criterion (an4.py): first-order change of log p(target) when a neuron is removed, using the original model's own
next-token distribution (pass 3, orig side) as baseline and both RMSNorm couplings (post_feedforward_layernorm and
model.norm). It now predicts the sign of E1 in all 7 text groups (predicted/observed ratio 0.0018-0.0041).

## Pass 4 (planned)
- Current edit E2: down_proj columns of 45 neurons in layers 13-17 set to 0 (list in work/edit.py, rule in
  work/sel_e2.py): first-order removal gain > 5 on (sv parallel + DEV lm A)/2, English gain >= -0.1 x that, and mean
  |activation| on Swedish >= 3x on English (they fire on Scandinavian text only). 28,800 weights (5 tensors).
  Sum of predicted gains: selection 442, DEV lm B (not used) 212, English +32 (unscaled units).
- Request: same 232-item probe-hidden-both as pass 3 (probe_both.py).
- Question: do these Scandinavian-only neurons really lower Swedish likelihood? Expect lower bpc on DEV lm A and B and
  parallel sv/da/no, English unchanged, small KL on English.
(Note: the pass-4 plan above was appended in the same step as the pass ran; its content was fixed before the pass.)

## Pass 4 result (probe hidden both, E2)
bpc DEV lm A -1.42% (selection set), DEV lm B +0.41% (not used for selection), SELF English -0.6%, parallel sv -0.9%,
da +1.4%, no +3.1%. KL per token on Swedish 0.19-0.26 (large), English 0.006. scala choice NLL 0.910 -> 0.747 but acc
0.633 -> 0.567. First-answer-token KL on Swedish prompts 0.17-0.20 (above the 0.1 target).
Diagnosis (an7.py): E2 raised entropy on Swedish text (2.26 -> 2.61 nats/token on DEV lm A); with the best logit scale
fitted afterwards for diagnosis only, NLL got worse (4.410 -> 4.457 A, 4.023 -> 4.109 B). So E2 is a flattening
(temperature) effect plus damage. Rejected.
Free findings: (1) the original model is strongly over-confident on raw text in every language: best logit scale 0.70
for Swedish AND English (NLL 4.85 -> 4.41 Swedish, 3.17 -> 2.85 English). General, not Swedish-specific; fixing it is a
global change of the output distribution, out of scope. (2) At Swedish function-word positions the model puts only
0.29% probability on the Danish/Norwegian form and 0.02% on the English form: no language confusion in likelihood.
(3) Top-1 errors on DEV lm are Swedish words (knowledge errors), not other languages.
New criterion an9.py: first-order removal gain split into a temperature part and a temperature-free part. It predicts
E2's DEV lm B temperature-free effect as -8 (observed: worse after rescaling), i.e. it agrees with pass 4.

## Pass 5 (planned)
- Current edit E3: down_proj columns of L15 n459, L17 n736, L17 n1067 set to 0 (1,920 weights, 2 tensors).
  Rule (sel_e3.py): Scandinavian-only (|act| Swedish >= 3x English) and temperature-free removal gain > 2 on each of sv
  parallel, DEV lm A, DEV lm B, English >= -1. L17 n1067 suppresses den/det/de/Den when positive.
- Request: same 232-item probe-hidden-both (probe_both.py), then an7.py for entropy and rescaled NLL.
- Question: does a small Swedish-only edit with no predicted temperature effect lower Swedish bpc on both DEV halves
  with small KL and no English change?

## Pass 5 result (probe hidden both, E3)
bpc DEV lm A -0.07%, DEV lm B -0.30%, parallel sv -1.20%, da -0.89%, no +2.66%, SELF English +0.03%, parallel en
-0.06%. KL per token: Swedish 0.014-0.017, English 0.0004. Entropy unchanged (2.26 -> 2.26 A); NLL after best rescaling
also lower (4.410 -> 4.407 A, 4.023 -> 4.017 B), so the gain is not a temperature effect. scala choice NLL
0.910 -> 0.876, acc 0.633 -> 0.600 (one item). First-answer-token KL: DEV gen prompts 0.032, SELF sv 0.056,
SELF en 0.0002. Small, consistent, Swedish-specific. The temperature-free criterion predicted the sign correctly.

## Pass 6 (planned)
- Current edit E4: down_proj columns set to 0 for 14 neurons: L15 n61, n459, n575; L16 n754; L17 n31, n52, n242,
  n262, n624, n736, n925, n1067, n1243, n1364 (8,960 weights, 3 tensors). Rule in sel_e4.py (relaxed E3 rule:
  |act| Swedish >= 2x English, temperature-free gain > 0 on sv parallel, DEV lm A and B with sum > 10, English >= -1).
- Request: same 232-item probe-hidden-both, then an7.py.
- Question: does the larger set give a larger Swedish gain on both DEV halves while keeping KL small and English
  unchanged? If it is not better than E3 on DEV lm B, keep E3.

## Pass 6 result (probe hidden both, E4)
Worse: bpc DEV lm A +0.80%, B +0.83%; entropy fell (2.26 -> 2.21), i.e. removal sharpened the distribution;
NLL after rescaling unchanged (A) / worse (B). English -0.25%. scala choice NLL 0.910 -> 0.987. Rejected.
an10.py: 9 of the 11 added neurons have a negative total first-order gain (their removal sharpens), which the relaxed
rule did not check.

## Pass 7 (planned)
- Current edit E5: E3 + L17 n1364 = down_proj columns of L15 n459, L17 n736, n1067, n1364 set to 0 (2,560 weights,
  2 tensors). Rule (sel_e5.py): Scandinavian-only (>= 2x), total AND temperature-free first-order gains > 0 on sv
  parallel, DEV lm A, DEV lm B, English >= -1. Exactly 4 neurons pass, a superset of E3.
- Request: same 232-item probe-hidden-both, then an7.py.
- Question: does adding n1364 improve on E3 on both DEV halves without raising KL much? Otherwise freeze E3.

## Pass 7 result (probe hidden both, E5)
Worse than E3: bpc DEV lm A +0.02%, B +0.16% (E3: -0.07%, -0.30%); KL per token on Swedish 0.05 (E3: 0.015).
Parallel sv -0.80%. Adding n1364 does not help. E3 is the best candidate.

## Freeze: E3 (work/edit.py FROZEN = "E3"); overlay rebuilt from the original weights.

## Pass 8 (planned)
- Current edit E3 (L15 n459, L17 n736, L17 n1067 down_proj columns = 0; 1,920 weights).
- Request: generate, which="both", DEV gen 30 + SELF sv prompts 30 + SELF en prompts 6 (same as pass 1).
- Question: does E3 change the language or repetition of greedy answers (sv_rate, repeat_rate), and do answers stay
  fluent? Expected: almost no change (first-token KL 0.03).

## Pass 8 result (generate both, E3)
DEV gen: sv_rate 0.733 -> 0.667 (one en answer became sv; three short sv answers became "other": "UFC 1488",
"O, min skål!", "Ingen bevis voor de exacte informatie"), repeat 0.100 -> 0.067, da/no unchanged (the Norwegian answer
stays Norwegian). SELF sv: sv 0.933 -> 0.900, repeat 0.033 -> 0.067. SELF en: all English. 43/66 greedy answers differ
somewhere in their 48 tokens (including 3 English ones), mostly after the first sentence. Quality is similar
(both often nonsensical). Changes are within noise for n=30; no sign of a language switch.

## Pass 9 (planned)
- Current edit E3 (frozen, unchanged).
- Request: score, which="both", with the harness scorer (bf16 lm_head, like final_eval): DEV lm 30, DEV scala 30,
  SELF English texts 12 (task lm_en_self), SELF Swedish parallel sentences 16 (task lm_par_sv).
- Question: does the official scorer reproduce the pass-5 bpc change on DEV lm and the English neutrality?

## Pass 9 result (score both, E3, harness scorer)
DEV lm bpc 1.9580 -> 1.9537 (-0.0043, -0.22%); 23/30 items better, 7 worse. scala acc 0.633 -> 0.600 (one item),
choice NLL 0.903 -> 0.858. SELF English bpc 0.9228 -> 0.9216 (-0.13%). Swedish parallel 2.0765 -> 2.0542 (-1.1%).
Matches pass 5. 1 pass left unused on purpose: no further candidate is justified by the evidence.

## Final: E3 frozen (edit hash b2de7462228fdec0). Next: sv/final_eval.py once.
