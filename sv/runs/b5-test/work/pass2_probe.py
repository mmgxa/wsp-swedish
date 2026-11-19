import sys, pickle, json
import numpy as np
sys.path.insert(0, "sv"); sys.path.insert(0, "sv/runs/b5-test/work")
import harness as H
from common import gen_items
from texts import PAR
r1 = pickle.load(open(sorted((H.run_dir() / "cache").glob("*.pkl"))[0], "rb"))
assert r1["kind"] == "generate"
gi = gen_items()
items, meta = [], []
for i, p in enumerate(PAR):
    for l in ("sv", "no", "da", "en"):
        items.append({"text": p[l]}); meta.append({"grp": f"par_{l}", "i": i})
for it in H.dev_items("lm", 30):
    items.append({"text": it["text"]}); meta.append({"grp": "lm_sv", "i": it["id"]})
tok = H.tokenizer()
for it, g in zip(gi, r1["orig"]["per_item"]):
    chat = H._chat(it["prompt"])
    assert chat.startswith("<bos>")
    n_prompt = len(H._ids(chat))  # tokens incl. bos
    items.append({"text": chat[len("<bos>"):] + g["text"]})
    meta.append({"grp": "gen_" + g["lang"], "i": it["id"], "n_prompt": n_prompt})
print(len(items), "items")
L = list(range(18))
r = H.fwd("probe", items, which="orig", what="mlp", layers=L, positions="all", note="pass2: mlp neuron language selectivity")
# compact statistics: per group, per layer: sum of act, sum of act^2, count, at content positions
groups = sorted({m["grp"] for m in meta})
S = {g: np.zeros((18, 2048)) for g in groups}; S2 = {g: np.zeros((18, 2048)) for g in groups}
P = {g: np.zeros((18, 2048)) for g in groups}; N = {g: 0 for g in groups}
per_item_mean = np.zeros((len(items), 18, 2048), np.float32)
for k, m in enumerate(meta):
    start = m.get("n_prompt", 1)  # skip bos (and the prompt for transcripts)
    for l in L:
        a = r["orig"]["mlp"][l][k][start:]
        S[m["grp"]][l] += a.sum(0); S2[m["grp"]][l] += (a ** 2).sum(0); P[m["grp"]][l] += (a > 0.05).sum(0)
        per_item_mean[k, l] = a.mean(0) if len(a) else 0
    N[m["grp"]] += len(r["orig"]["mlp"][0][k]) - start
np.savez_compressed(H.run_dir() / "work" / "pass2_stats.npz", groups=np.array(groups),
                    S=np.stack([S[g] for g in groups]), S2=np.stack([S2[g] for g in groups]),
                    P=np.stack([P[g] for g in groups]), N=np.array([N[g] for g in groups]), item_mean=per_item_mean)
json.dump(meta, open(H.run_dir() / "work" / "pass2_meta.json", "w"))
print({g: N[g] for g in groups})
