building edited model ...
  lm         n=  49 (3s)
  scala      n=  50 (1s)
  gen        n=  50 (3s)

b2-test: 2/2 passes; 0.0045% of weights changed (5 tensors)
task          n  metric           orig     edit    delta  95% CI of delta
lm           49  bits/char       1.896    1.989   +0.093  [+0.084, +0.102]
scala        50  acc             0.560    0.560   +0.000  [+0.000, +0.000]
                 choice_nll      1.072    1.143   +0.071  [+0.031, +0.112]
gen          50  sv_rate         0.580    0.520   -0.060  [-0.140, +0.000]
gen          50  repeat_rate     0.040    0.060   +0.020  [-0.040, +0.100]
kl_generic_prompts_nats: 0.0148
langdetect (independent): {'orig': {'sv': 0.58, 'en': 0.1, 'da': 0.0, 'no': 0.06, 'other': 0.26}, 'edit': {'sv': 0.58, 'en': 0.08, 'da': 0.0, 'no': 0.04, 'other': 0.3}}
