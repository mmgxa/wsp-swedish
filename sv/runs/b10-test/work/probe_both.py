"""One counted probe pass (final hidden states, orig and current edit) + analysis. Usage: probe_both.py TAG"""
import sys, pickle, numpy as np; sys.path.insert(0, "sv"); sys.path.insert(0, "sv/runs/b10-test/work")
import harness as H, probe_lib as P
tag = sys.argv[1]
its = P.build_items()
r = H.fwd("probe", [it["req"] for it in its], which="both", what="hidden", layers=[17], positions="all",
          note=f"{tag}: final hidden both, bpc/scala/KL")
hid = {s: [a.astype(np.float16) for a in r[s]["hidden"][17]] for s in ("orig", "edit")}
pickle.dump({"items": its, "tokens": r["orig"]["tokens"], "hid": hid, "edit_hash": H.edit_hash()},
            open(f"sv/runs/b10-test/work/out/{tag}.pkl", "wb"))
per, kl = P.analyse(its, r["orig"]["tokens"], hid)
out = P.summary(its, per, kl); P.print_summary(out)
pickle.dump({"per": per, "kl": kl, "out": out}, open(f"sv/runs/b10-test/work/out/{tag}_an.pkl", "wb"))
