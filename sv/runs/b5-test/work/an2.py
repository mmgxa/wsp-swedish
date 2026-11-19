import sys, numpy as np, torch
from collections import Counter
sys.path.insert(0, "sv"); sys.path.insert(0, "sv/runs/b5-test/work")
import harness as H
from texts import PAR
tok = H.tokenizer()
cnt = {l: Counter() for l in ("sv", "no", "da", "en")}
for p in PAR:
    for l in cnt: cnt[l].update(H._ids(p[l]))
for it in H.dev_items("lm", 30): cnt["sv"].update(H._ids(it["text"]))
noda = Counter(); noda.update(cnt["no"]); noda.update(cnt["da"])
T_noda = [t for t, c in noda.items() if c >= 2 and cnt["sv"][t] == 0]
T_sv = [t for t, c in cnt["sv"].items() if c >= 3 and noda[t] == 0]
T_en = [t for t, c in cnt["en"].items() if c >= 2 and cnt["sv"][t] == 0 and noda[t] == 0]
print(len(T_noda), len(T_sv), len(T_en))
print("noda:", [tok.decode([t]) for t in sorted(T_noda, key=lambda t: -noda[t])[:40]])
print("sv:", [tok.decode([t]) for t in sorted(T_sv, key=lambda t: -cnt['sv'][t])[:40]])
np.savez(H.run_dir() / "work/tokensets.npz", noda=np.array(T_noda), sv=np.array(T_sv), en=np.array(T_en))
