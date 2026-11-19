# Explanations (plain English, one entry per step)

## Step 0: plan
Budget is 10 passes, DEV is 30 items per task (lm, scala, gen). With so little data I fix the DEV set to the harness
DEV items and add hand-written items (SELF) for diagnosis. First I look at the original model's Swedish generations,
because a switchable failure (wrong language, repetition) is the kind of thing the blog found editable, and only a
generation pass can show it. If the model already answers in Swedish almost always and does not loop, the language
hypothesis is refuted and I look for a likelihood gain instead.

## Step 1: baseline generations (pass 1)
The original model already answers Swedish prompts in Swedish most of the time (22/30 DEV, 28/30 SELF). Only one DEV
answer is Norwegian and one is English (an English song title echoed back), so wrong-language answers are rare and a
"language switch" edit has little room to help on gen. Repetition occurs in 3/30 DEV answers. The largest oddity is
the glitch token "ovce" at the start of nearly every answer; it occurs exactly when the prompt is left-padded inside
the batched generation (unpadded prompts never start with it), so it is an interaction between the harness's batched
padding and the model, present for both models and on HOLDOUT as well. I note it, but it is not a Swedish skill.
Next I look inside: one probe pass of all MLP neuron activations on Swedish, English, Danish and Norwegian text, to find
neurons that push Swedish text towards other languages or against the correct next token. If no neuron has a
consistent harmful direct effect on Swedish text, the "harmful component" hypothesis is refuted.

## Step 2: neuron activations (pass 2)
Among all 36,864 MLP neurons, the one with the largest direct harm to the correct next token on Swedish text, while
helping English, is neuron 1977 in the last layer (L17). It is active at almost the same negative value at every
position in every language, so it behaves like a constant logit bias, and that bias favours English words over Swedish
ones. The harm also appears on DEV lm items 15-29 that I did not use for the ranking, which makes it less likely to be
a selection artefact. The direct-effect number ignores competition between tokens, so it does not predict the change
in log-likelihood. Next I remove the neuron and compare orig and edited in one probe pass of final hidden states, which
gives full next-token distributions (bpc, scala, KL) for both models. If Swedish bpc does not go down, or goes down only
as much as English goes up, the hypothesis that this neuron limits Swedish is refuted.

## Step 3: E1 result (pass 3)
Removing L17 n1977 made every language worse by 5-9% bpc, so the neuron is a useful constant bias and my first
ranking, which compared the correct token with the average of all 262k vocabulary embeddings, was misleading: the
neuron mostly pushes down junk tokens, which helps every language. With pass 3 I now have the original model's full
next-token distribution, so I re-ranked neurons by the first-order change in log p(correct token) when a neuron is
removed, measured against the model's own prediction and including both RMSNorm couplings. That criterion correctly
predicts E1's harm in all text groups. Among late-layer neurons it finds a group that fire only on Scandinavian text
(near zero on English) and whose output lowers the probability of the correct Swedish token. I remove 45 of them
(E2), chosen by fixed rules on Swedish parallel sentences and DEV lm items 0-14. If Swedish bpc does not drop on
DEV lm 15-29, or English changes, the hypothesis is refuted.

## Step 4: E2 result (pass 4) and what it revealed
E2 helped the DEV half it was chosen on (-1.4% bpc) and hurt the other half (+0.4%), with a large KL on Swedish text.
Looking at entropy explained why: the edit made Swedish predictions flatter, and the original model is strongly
over-confident on raw text (its NLL would drop 9% in Swedish and 10% in English if its logits were simply scaled by
0.7). My first-order neuron ranking was mostly finding neurons whose removal flattens the distribution. That is a
general calibration issue of this instruction-tuned model, the same in English and Swedish, so it is not a Swedish
obstacle, and fixing it would be a global change of the output distribution, which is out of scope. I also checked
the language-confusion hypothesis directly: at Swedish function-word positions the Danish/Norwegian form gets 0.3% of
the probability and English 0.02%, so language mixing does not limit Swedish likelihood. I now separate the
temperature part from the rest in the ranking. Only three Scandinavian-only neurons have a temperature-free gain on all
three Swedish groups; E3 removes them. If DEV lm A and B do not both improve, the remaining "harmful Swedish neuron"
hypothesis is refuted too.

## Step 5: E3 result (pass 5)
Removing the three neurons lowered Swedish bpc slightly on both DEV halves (-0.07% and -0.30%) and on the Swedish
parallel sentences (-1.2%), left English unchanged (KL 0.0004 per token), and did not change entropy, so for the
first time the gain is not a flattening effect. The effect is small. Because the temperature-free criterion predicted
it correctly, I try a relaxed version of the same rule (14 neurons, E4). If E4 is not better than E3 on DEV lm 15-29
or raises the KL clearly, I keep E3.

## Step 6: E4 result (pass 6)
E4 made Swedish worse on both halves (+0.8%) by sharpening the distribution, which hurts an already over-confident
model. Most of the added neurons have a negative total first-order gain; my relaxed rule only checked the
temperature-free part. The corrected rule requires both parts to be positive on all three Swedish groups. Exactly four
neurons pass: the three of E3 plus L17 n1364, a neuron that fires only on Scandinavian text. E5 tests them. If E5 is
not better than E3 on DEV lm 15-29, I freeze E3.

## Step 7: E5 result (pass 7) and freeze
Adding n1364 to E3 removed the gain on DEV lm (+0.02% and +0.16%) and tripled the KL, so E3 stays the best edit. I
freeze E3 now, before any HOLDOUT use. The remaining passes check E3 on generation (pass 8) and with the harness's own
teacher-forced scorer (pass 9), which computes logits in bf16 like the final evaluation. If generation shows more
non-Swedish or repeated answers, I will report it as a cost of the edit.

## Step 8: generation check (pass 8)
E3 does not change which language the model answers in: the one Norwegian answer stays Norwegian, one English echo
became Swedish, and three short Swedish answers became too short or odd to be labelled Swedish, so sv_rate moved from
0.733 to 0.667 on 30 DEV prompts. Repetition went down on DEV (3 to 2) and up on SELF (1 to 2). With 30 prompts a
change of two answers is noise, and many answers change wording because greedy decoding diverges at near-ties. I keep
E3 frozen because the generation evidence is neutral and the likelihood evidence is consistent. Pass 9 checks the bpc
with the harness's own scorer. If the scorer shows no gain or an English cost, I will report that.

## Step 9: harness scorer (pass 9), HOLDOUT, conclusion
The harness scorer confirmed the DEV result: Swedish bpc 1.9580 -> 1.9537 (23 of 30 texts better), English unchanged
or slightly better, scala choice NLL lower and accuracy one item lower. I left the last pass unused because no further
candidate was justified. On HOLDOUT (final_eval.py, run once on the frozen E3), Swedish bpc changed by -0.001 with a 95%
interval [-0.003, +0.002] that includes zero, so the DEV gain did not replicate at a detectable size. scala choice NLL
improved (-0.045, interval [-0.078, -0.015]) while accuracy fell by one item. sv_rate stayed at 0.58 (langdetect 0.58 ->
0.60). Repetition rose from 2 to 5 of 50 answers (interval includes zero). KL on generic prompts was 0.015 nats.
Conclusion: I found no switchable Swedish failure in this model. Wrong-language answers are rare, and Danish,
Norwegian and English forms get almost no probability inside Swedish text. The largest effect I measured is general
over-confidence on raw text, which is out of scope. The submitted three-neuron edit is a small, Swedish-only
likelihood change that is not significant on HOLDOUT.
