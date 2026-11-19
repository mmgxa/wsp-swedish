# Explanations (plain English, one entry per step)

## Step 1 (pass 1, generate orig)
The original model answers 67% of the 30 DEV questions in Swedish by the heuristic (94% of my 34 simple prompts).
The failures are a switch of language: Norwegian or Danish openings ("Det er vanskelig å si", "Jeg er ikke sikker")
and English answers when the question contains an English song or film title. There is no repetition at 48 tokens.
So the candidate switchable failure is "the model leaves Swedish", and repetition is ruled out. Next I looked for
the components that encode "this is Swedish" with one MLP probe on parallel text in four languages.
A refutation would have been a probe showing no Swedish-specific components.

## Step 2 (pass 2, probe MLP activations on parallel sv/no/da/en text and on the pass-1 transcripts)
Language-specific neurons sit in layers 13-17, plus character-level ones in layer 0. Two neurons stand out:
L14 n1542 and L13 n72 fire only on Swedish (about +1.3 to +1.5 on Swedish, below 0 on the other three languages).
Read through the unembedding, their output columns push Swedish-only tokens up against both Norwegian/Danish-only
and English-only tokens. They also look like a switch: at the last prompt token, before any answer token exists,
they are much weaker for prompts the model then answers in Norwegian, Danish or English. This is still
correlational, because the prompt itself may differ. The causal test is to amplify the two output columns (x2) and
see whether Swedish likelihood improves and whether answers stay Swedish more often. I score first because a score
pass covers Swedish, Norwegian, Danish and English text at once. If Swedish bpc rises, or English bpc rises by more
than about 1%, the hypothesis "more of this neuron = better Swedish" is refuted at this strength.

## Step 3 (pass 3, score: Swedish neurons x2)
Doubling the two Swedish neurons made Swedish text less likely, not more: DEV bits per char rose from 1.957 to 2.010
and only 8 of 30 texts improved. Norwegian and Danish got worse by a similar amount and English barely moved. So the
neurons are Swedish-specific (English is untouched) but the model's Swedish likelihood already sits near their
current strength, and pushing harder distorts the predictions. scala accuracy also fell by two items. I drop
amplification. With two passes left I cannot run a dose-response; instead I take the edit from the other side of the
same finding. The original model's failures are drifts into Norwegian or Danish, and there are two late neurons that
fire only on Norwegian (L15 n459) or Danish (L14 n1594), are silent on Swedish and English, and write toward
Norwegian/Danish tokens. L15 n459 fires exactly where the drift answers turn Norwegian. Removing them should cost
nothing on Swedish (they are silent there) and should make Norwegian/Danish text less likely. If Swedish or English
bpc rises noticeably, or Norwegian/Danish bpc does not rise, the "language neuron" reading is wrong.

## Step 4 (pass 4, score: Norwegian/Danish neurons removed)
Removing the two neurons did what the reading predicted. Norwegian and Danish text became less likely (+0.029 and
+0.037 bits per char), Swedish became slightly more likely (-0.003 on DEV lm, 21 of 30 texts better; -0.0035 on my
Swedish paragraphs) and English barely moved (+0.0026, 0.26%). scala accuracy is unchanged. The Swedish gain is small,
which fits neurons that are almost silent on Swedish: they only leak a little Norwegian/Danish probability there.
The main expected effect is on the switch: answers that the original model started in Norwegian or Danish.
The last pass generates with both models on the same 64 prompts as pass 1. If the drift answers stay Norwegian/Danish,
the neurons are a marker of the drift and not its cause; if previously Swedish answers break, the edit has a cost.
I will freeze edit B after this pass regardless, since no pass remains to test an alternative.

## Step 5 (pass 5, generate: Norwegian/Danish neurons removed) and freeze
On DEV, one real switch happened: the Norwegian answer "Det er vanskelig å si ..." became Swedish ("Det är en fråga
som ..."). A second item moved from en to sv by the label only; its edited text is a repeated Danish-looking line,
which is also the one new repetition. One Danish answer stayed Danish. The 34 simple prompts kept 94% Swedish, and 41
of 64 answers were identical. So the edit removes some of the Norwegian push, but most drift is not caused by these two
neurons alone. No pass remained, so I froze edit B and ran final_eval.py once.

## Step 6 (HOLDOUT, final_eval.py, run once on edit B)
HOLDOUT shows essentially no effect: Swedish bits per char 1.896 -> 1.895 (-0.001, 95% CI [-0.002, +0.000]), gen
sv_rate 0.58 -> 0.58, scala accuracy 0.56 -> 0.56 but scala choice_nll worse by +0.034 (CI [+0.013, +0.057]), KL on
generic prompts 0.0012 nats. The DEV gain in language switching did not reach the holdout. The honest conclusion: these
neurons are real Norwegian/Danish detectors (removing them raises Norwegian/Danish bpc by 2-2.5%), but the drifts in the
model's answers are not switched by them alone, and the effect on Swedish is at the size of noise.
