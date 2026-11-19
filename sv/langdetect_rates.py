"""Independent language check of generations with `langdetect` (evaluation environment).

    .venv-eval/bin/python sv/langdetect_rates.py in.json out.json
in.json: {"orig": [text, ...], "edit": [text, ...]}  ->  out.json: share of texts detected as sv / en / da / no / other.
"""
import json
import sys
from collections import Counter

from langdetect import DetectorFactory, detect

DetectorFactory.seed = 0
d = json.load(open(sys.argv[1]))
out = {}
for side, texts in d.items():
    labs = []
    for t in texts:
        try:
            labs.append(detect(t) if len(t.split()) >= 3 else "other")
        except Exception:
            labs.append("other")
    c = Counter(labs)
    n = max(len(labs), 1)
    out[side] = {"n": len(labs), "sv": c["sv"] / n, "en": c["en"] / n, "da": c["da"] / n, "no": c["no"] / n,
                 "other": (n - c["sv"] - c["en"] - c["da"] - c["no"]) / n}
json.dump(out, open(sys.argv[2], "w"))
