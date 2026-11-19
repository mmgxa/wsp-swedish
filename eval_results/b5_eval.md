building edited model ...
  lm         n=  49 (3s)
  scala      n=  50 (1s)
  gen        n=  50 (3s)

b5-test: 5/5 passes; 0.0005% of weights changed (2 tensors)
task          n  metric           orig     edit    delta  95% CI of delta
lm           49  bits/char       1.896    1.895   -0.001  [-0.002, +0.000]
scala        50  acc             0.560    0.560   +0.000  [+0.000, +0.000]
                 choice_nll      1.072    1.106   +0.034  [+0.013, +0.057]
gen          50  sv_rate         0.580    0.580   +0.000  [+0.000, +0.000]
gen          50  repeat_rate     0.040    0.040   +0.000  [+0.000, +0.000]
kl_generic_prompts_nats: 0.0012
langdetect (independent): {'orig': {'sv': 0.58, 'en': 0.1, 'da': 0.0, 'no': 0.06, 'other': 0.26}, 'edit': {'sv': 0.58, 'en': 0.08, 'da': 0.0, 'no': 0.06, 'other': 0.28}}
