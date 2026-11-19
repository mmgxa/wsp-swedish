building edited model ...
  lm         n=  49 (3s)
  scala      n=  50 (1s)
  gen        n=  50 (3s)

b10-test: 9/10 passes; 0.0007% of weights changed (2 tensors)
task          n  metric           orig     edit    delta  95% CI of delta
lm           49  bits/char       1.896    1.895   -0.001  [-0.003, +0.002]
scala        50  acc             0.560    0.540   -0.020  [-0.060, +0.000]
                 choice_nll      1.072    1.027   -0.045  [-0.078, -0.015]
gen          50  sv_rate         0.580    0.580   +0.000  [-0.060, +0.060]
gen          50  repeat_rate     0.040    0.100   +0.060  [-0.020, +0.160]
kl_generic_prompts_nats: 0.0146
langdetect (independent): {'orig': {'sv': 0.58, 'en': 0.1, 'da': 0.0, 'no': 0.06, 'other': 0.26}, 'edit': {'sv': 0.6, 'en': 0.1, 'da': 0.0, 'no': 0.06, 'other': 0.24}}
