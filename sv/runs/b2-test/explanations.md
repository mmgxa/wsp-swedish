# Explanations (plain English, one entry per step)

## Step 1: weight reading (free)

What I did: built a Swedish-vs-English direction from the embedding matrix and measured how much each layer's
attention output and MLP output can write along it.
What it showed: nothing stands out. Every component writes along this direction less than a random direction would.
Late layers have very large post-norm gains (post_feedforward gain about 300 in layer 17), so late MLPs dominate the
residual stream and are the natural place to look for direct effects on the output.
Conclusion: the weights alone do not reveal a component that is "against Swedish". With only 2 passes, I spend pass 1 on
reading MLP activations of the original model on Swedish and English text, which gives neuron-level resolution, and
keep pass 2 for one paired test of the edit.
Next: pass 1, MLP probe on 70 Swedish + 50 English texts (DEV material only; the 30 tiny-DEV `lm` items are kept out).
What would refute the plan: if no neuron has a harmful direct effect on Swedish next tokens that replicates between the
two halves of the Swedish texts, there is no neuron-level obstacle to remove and I will report that.

## Step 2: pass 1 (MLP activations of the original model)

What the last result showed: I computed, for every MLP neuron, its direct contribution to the logit of the actual next
token in 70 Swedish and 50 English texts. Centred on the vocabulary mean, the most "harmful" neurons were large
calibration neurons that push all common tokens down in both languages (they hurt English as much as Swedish), so I
re-centred on the texts' own unigram distribution, which cancels most of that shared effect. After that, a small set of
late-layer neurons (layers 12-17) still lowers the true Swedish next token on 80-100% of the Swedish texts, equally in
both halves of the texts (half-to-half correlation of the per-neuron scores 0.97), while doing almost nothing on
English. Reading their output vectors against the embedding: L17 n798 promotes Spanish/Italian words, L16 n999
promotes English content words and suppresses Germanic endings such as -kt, -ks, -cht; L13 n1771 suppresses English
function words but promotes Hungarian/Portuguese fragments. So the picture is: some neurons fire on Swedish text and
push toward other languages' tokens.
Conclusion: removing 19 such neurons should lower Swedish bits per character a little, with little effect on English.
Direct attribution ignores the effects these neurons have through later layers and through the final norm, so the true
effect could be smaller or even reversed. This is the main risk.
Next: pass 2, one paired score pass (orig and edit on the same items), with 30 Swedish DEV texts the selection never
saw as the clean test, plus English and multiple-choice retention checks.
What would refute the hypothesis: Swedish bpc on the 30 unseen DEV texts not lower (or lower by less than noise), or
English bpc / multiple-choice choice_nll clearly worse.

## Step 3: pass 2 (paired test of the edit)

What the last result showed: removing the 19 neurons made Swedish text less likely, not more: bits per character rose
by 0.09 (4.6%) on the 30 unseen DEV texts, with 29 of 30 texts worse, and by a similar amount on the 70 texts used to
pick the neurons. English got 1.5% worse. On ScaLA the correct-answer loss got worse; on four general multiple-choice
tasks it got better by about 0.5 nats while accuracy did not move, which looks like a change in confidence, not in skill.
Conclusion: the hypothesis is refuted. The direct-effect score was wrong about the sign, even in-sample, so these
neurons' net effect on Swedish runs through later layers and the final norm (for example by setting the residual
scale), which a direct read of the logit cannot see. Pass 1 found no switchable "anti-Swedish" component that this
method can identify.
Next: no passes are left, so I cannot test a different edit. I keep the rule I fixed before the pass: freeze this edit
and run the holdout once, then report it as a failure. Inverting the edit (amplifying the neurons) would be an untested
guess based on one result, so I do not submit it.
What would change this conclusion: a holdout result where Swedish bpc does not get worse would contradict DEV; I expect
it to get worse by about 0.08-0.09 bpc.

## Step 4: holdout (final_eval.py, run once on the frozen edit)

What it showed: the same as DEV. Swedish bpc +0.093 (CI [+0.084, +0.102]), scala choice_nll +0.071, gen sv_rate
0.58 -> 0.52 by the heuristic and unchanged by langdetect, KL on generic prompts 0.015 nats.
Conclusion: the edit is a reproducible small harm to Swedish, not a gain. Reported as a failure in REPORT.md.
