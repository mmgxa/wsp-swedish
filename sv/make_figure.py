"""Results figure from the three runs: uv run --offline python sv/make_figure.py  ->  figures/results.png"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RUNS = {"b2": 2, "b5": 5, "b10": 10}
rows = []
for r, budget in RUNS.items():
    d = json.loads((ROOT / f"sv/runs/{r}-test/results.json").read_text())
    t = d["tasks"]
    log = [json.loads(x) for x in (ROOT / f"sv/runs/{r}-test/forward_log.jsonl").read_text().splitlines()]
    rows.append({"run": f"{budget} passes", "used": sum(x.get("type") == "pass" for x in log), "budget": budget,
                 "pct": d["edit"]["pct_changed"], "n": d["edit"]["changed_weights"],
                 "lm": t["lm"]["delta_bpc"], "lm_lo": t["lm"]["delta_bpc_ci"][0], "lm_hi": t["lm"]["delta_bpc_ci"][1],
                 "sc": t["scala"]["delta_choice_nll"], "sc_lo": t["scala"]["delta_choice_nll_ci"][0], "sc_hi": t["scala"]["delta_choice_nll_ci"][1],
                 "kl": d["kl_generic_prompts_nats"]})
df = pd.DataFrame(rows).set_index("run")
g = {}
for r, budget in RUNS.items():
    t = json.loads((ROOT / f"sv/runs/{r}-test/results.json").read_text())["tasks"]["gen"]
    g[f"{budget} passes"] = {"orig": t["orig"]["sv_rate"], "edit": t["edit"]["sv_rate"],
                             "orig_ci": t["sv_rate_ci"]["orig"], "edit_ci": t["sv_rate_ci"]["edit"]}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .25, "axes.axisbelow": True})
C = {"2 passes": "#c0504d", "5 passes": "#4f81bd", "10 passes": "#2e7d6b"}
fig, ax = plt.subplots(2, 2, figsize=(11, 8))


def delta(a, col, lo, hi, title, xlabel):
    y = range(len(df))
    for i, (run, r) in enumerate(df.iterrows()):
        a.errorbar(r[col], i, xerr=[[r[col] - r[lo]], [r[hi] - r[col]]], fmt="o", color=C[run], capsize=4, ms=8, lw=2)
        a.annotate(f"{r[col]:+.3f}", (r[hi], i), xytext=(6, 9), textcoords="offset points", va="center", fontsize=9)
    a.axvline(0, color="k", lw=1)
    a.set_yticks(list(y)); a.set_yticklabels(df.index); a.invert_yaxis()
    a.set_title(title, loc="left", fontweight="bold"); a.set_xlabel(xlabel)
    lim = max(abs(df[lo]).max(), abs(df[hi]).max()) * 1.35
    a.set_xlim(-lim, lim)


delta(ax[0, 0], "lm", "lm_lo", "lm_hi", "A. Swedish text: bits per character", "edited − original (lower = better), 95% CI")
ins = ax[0, 0].inset_axes([.04, .12, .36, .48])
for i, (run, r) in enumerate(df.iterrows()):
    if run != "2 passes":
        ins.errorbar(r["lm"], i, xerr=[[r["lm"] - r["lm_lo"]], [r["lm_hi"] - r["lm"]]], fmt="o", color=C[run], capsize=3, ms=6, lw=2)
ins.axvline(0, color="k", lw=1); ins.set_xlim(-.004, .004); ins.set_ylim(2.6, .4)
ins.set_yticks([1, 2]); ins.set_yticklabels(["5", "10"], fontsize=8); ins.tick_params(axis="x", labelsize=8)
ins.set_title("zoom (x ±0.004): 5- and 10-pass runs", fontsize=8, loc="left"); ins.grid(alpha=.25)
delta(ax[0, 1], "sc", "sc_lo", "sc_hi", "B. ScaLA-sv: choice NLL", "edited − original (lower = better), 95% CI")

a = ax[1, 0]
w = .36
for i, (run, v) in enumerate(g.items()):
    for k, (key, col) in enumerate([("orig", "#b9b9b9"), ("edit", C[run])]):
        x = i + (k - .5) * w
        a.bar(x, v[key], w, color=col, label=("original" if key == "orig" else "edited") if i == 0 else None)
        lo, hi = v[key + "_ci"]
        a.errorbar(x, v[key], yerr=[[v[key] - lo], [hi - v[key]]], color="k", capsize=3, lw=1)
        a.text(x, hi + .02, f"{v[key]:.2f}", ha="center", fontsize=8)
a.set_xticks(range(len(g))); a.set_xticklabels(list(g)); a.set_ylim(0, .95)
a.set_ylabel("share of answers labelled Swedish")
a.set_title("C. Generation: language of answers", loc="left", fontweight="bold")
a.legend(frameon=False, loc="upper right", fontsize=9, handles=[plt.Rectangle((0, 0), 1, 1, color="#b9b9b9"), plt.Rectangle((0, 0), 1, 1, color="#6b6b6b")], labels=["original", "edited"])

a = ax[1, 1]
x = range(len(df))
a.bar(x, df["budget"], color="#dddddd", label="budget", width=.6)
a.bar(x, df["used"], color=[C[r] for r in df.index], width=.6, label="used")
for i, (run, r) in enumerate(df.iterrows()):
    a.text(i, r["budget"] + .3, f"{int(r['n']):,} weights\n({r['pct']:.4f}%)\nKL {r['kl']:.4f}", ha="center", fontsize=8)
a.set_xticks(list(x)); a.set_xticklabels(df.index); a.set_ylim(0, 15)
a.set_ylabel("forward passes"); a.set_title("D. Passes used and size of the edit", loc="left", fontweight="bold")

fig.suptitle("Weight-space programming for Swedish on Gemma-3-270M-it: no run improved Swedish text or answer language\n"
             "holdout = EuroEval test split, n ≈ 50 per task, one run per budget", fontsize=12, y=.995)
fig.tight_layout(rect=(0, 0, 1, .96))
out = ROOT / "figures"; out.mkdir(exist_ok=True)
fig.savefig(out / "results.png", dpi=170)
print(out / "results.png"); print(df.round(4).to_string())
