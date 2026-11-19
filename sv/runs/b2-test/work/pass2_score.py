"""Pass 2: paired score pass (orig vs edit) on DEV material. Saves per-item results to work/pass2.pkl."""
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "sv"))
import harness as H  # noqa: E402

DEF = ROOT / "sv/data/default/dev"
tiny_lm = H.dev_items("lm")
tiny_ids = {r["id"] for r in tiny_lm}
sel_lm = [dict(r, task="lm_sel") for r in H.load_jsonl(DEF / "lm.jsonl") if r["id"] not in tiny_ids]
lm_en = H.load_jsonl(DEF / "lm_en.jsonl")[50:]
scala_t = H.dev_items("scala")
st_ids = {r["id"] for r in scala_t}
scala = scala_t + [r for r in H.load_jsonl(DEF / "scala.jsonl") if r["id"] not in st_ids]
mc = [r for t in ("swerec", "hellaswag", "arc", "mmlu", "belebele") for r in H.load_jsonl(DEF / f"{t}.jsonl")]
items = tiny_lm + sel_lm + lm_en + scala + mc
print("items", len(items), {t: sum(i["task"] == t for i in items) for t in dict.fromkeys(i["task"] for i in items)})

res = H.fwd("score", items, which="both", note="edit 19 neurons zeroed: tiny-DEV lm (clean), lm_sel, lm_en(50-99), scala 100, 5 mc x100")
H.show(res)
pickle.dump({"items": items, "res": res}, open(Path(__file__).with_name("pass2.pkl"), "wb"))

# paired per-text bpc differences for the lm tasks (bootstrap over texts, seed 0)
rng = np.random.default_rng(0)
for t in ("lm", "lm_sel", "lm_en"):
    idx = [k for k, i in enumerate(items) if i["task"] == t]
    no = np.array([res["orig"]["per_item"][k]["nll"] for k in idx]); ne = np.array([res["edit"]["per_item"][k]["nll"] for k in idx])
    nc = np.array([items[k]["text"].__len__() for k in idx], float)
    f = lambda i: (ne[i].sum() - no[i].sum()) / nc[i].sum() / np.log(2)
    bs = [f(rng.integers(0, len(idx), len(idx))) for _ in range(2000)]
    better = np.mean((ne / nc) < (no / nc))
    print(f"{t:7s} n={len(idx)} delta bpc {f(np.arange(len(idx))):+.4f}  95% CI [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]  "
          f"texts improved {better:.2f}  rel {100 * (ne.sum() / no.sum() - 1):+.2f}%")
