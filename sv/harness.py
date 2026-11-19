"""Harness for Weight-Space Programming on gemma-3-270m-it (Swedish).

The editor (an LLM agent) may run the model ONLY through `fwd()`. Each call is one counted pass.
Analysis of the weights is free. Edits are stored as an overlay of replaced tensors via `save_edit()`.

Typical use, from `uv run python`:

    import sys; sys.path.insert(0, "sv")
    import harness as H
    H.status()                                    # budget, passes left, edit summary
    items = H.dev_items("lm", 50) + H.dev_items("scala", 50)
    res = H.fwd("score", items, which="both")     # ONE pass
    H.show(res)                                   # compact table: orig vs edit

Run directory: $WSP_RUN_DIR, else the path in sv/runs/CURRENT.
"""
from __future__ import annotations

import contextlib
import copy
import fcntl
import hashlib
import json
import math
import os
import pickle
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parent.parent
SV = ROOT / "sv"
MODEL_DIR = ROOT / "gemma-3-270m-it"
FORBIDDEN_TENSORS = {"model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class BudgetExhausted(RuntimeError):
    pass


class HarnessError(RuntimeError):
    pass


# --------------------------------------------------------------------------- run directory
def run_dir() -> Path:
    p = os.environ.get("WSP_RUN_DIR")
    if not p:
        cur = SV / "runs" / "CURRENT"
        if not cur.exists():
            raise HarnessError("no run: set WSP_RUN_DIR or run sv/init_run.py")
        p = cur.read_text().strip()
    return Path(p)


def config() -> dict:
    return json.loads((run_dir() / "config.json").read_text())


def data_dir() -> Path:
    return SV / "data" / config()["data"] / "dev"


def edit_dir() -> Path:
    return Path(os.environ.get("WSP_EDIT_DIR") or (run_dir() / "edit"))


def _log_path() -> Path:
    return run_dir() / "forward_log.jsonl"


def passes_used() -> int:
    p = _log_path()
    if not p.exists():
        return 0
    return sum(1 for ln in p.read_text().splitlines() if ln.strip() and json.loads(ln).get("type") == "pass")


def status() -> dict:
    c = config()
    used = passes_used()
    st = {"run": run_dir().name, "budget": c["budget"], "used": used, "remaining": c["budget"] - used,
          "max_items_per_pass": c["max_items_per_pass"], "edit_hash": edit_hash(), "edit_tensors": len(_overlay_names())}
    print(json.dumps(st))
    return st


# --------------------------------------------------------------------------- data
def dev_items(task: str, n: int | None = None, offset: int = 0) -> list[dict]:
    """DEV items of one task (free to read). Tasks: see config()['tasks']."""
    p = data_dir() / f"{task}.jsonl"
    rows = [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]
    rows = rows[offset:]
    return rows if n is None else rows[:n]


# --------------------------------------------------------------------------- weights and edits
_orig_file = None


def _orig():
    global _orig_file
    if _orig_file is None:
        _orig_file = safe_open(str(MODEL_DIR / "model.safetensors"), "pt")
    return _orig_file


def tensor_names() -> list[str]:
    return list(_orig().keys())


def orig_tensor(name: str) -> torch.Tensor:
    """Original weight as float32 on CPU (a fresh copy)."""
    return _orig().get_tensor(name).float().clone()


def _overlay_path() -> Path:
    return edit_dir() / "overlay.safetensors"


def _overlay() -> dict[str, torch.Tensor]:
    p = _overlay_path()
    return load_file(str(p)) if p.exists() else {}


def _overlay_names() -> list[str]:
    p = _overlay_path()
    if not p.exists():
        return []
    with safe_open(str(p), "pt") as f:
        return list(f.keys())


def edited_tensor(name: str) -> torch.Tensor:
    """Current value of a tensor (float32): the edit if there is one, else the original."""
    ov = _overlay()
    return (ov[name] if name in ov else orig_tensor(name)).float().clone()


def edit_hash() -> str:
    p = _overlay_path()
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() else "none"


def reset_edit() -> None:
    if _overlay_path().exists():
        _overlay_path().unlink()


def save_edit(name: str, tensor: torch.Tensor) -> dict:
    """Replace tensor `name` by `tensor` (float32 is fine; the model stores bf16).

    Refused: embed_tokens / lm_head / final norm, wrong name or shape, non-finite values,
    and a whole-tensor multiplication by a constant. Free (no forward pass)."""
    if name in FORBIDDEN_TENSORS:
        raise HarnessError(f"{name} may not be edited")
    if name not in tensor_names():
        raise HarnessError(f"unknown tensor {name}")
    o = orig_tensor(name)
    t = torch.as_tensor(tensor).detach().float().cpu()
    if t.shape != o.shape:
        raise HarnessError(f"shape {tuple(t.shape)} != {tuple(o.shape)}")
    if not torch.isfinite(t).all():
        raise HarnessError("non-finite values")
    r = float((t * o).sum() / (o * o).sum().clamp_min(1e-30))
    if abs(r - 1) > 1e-6 and float((t - r * o).norm()) <= 1e-4 * float(t.norm()):
        raise HarnessError(f"whole-tensor multiplication by a constant ({r:.4f})")
    ov = _overlay()
    ov[name] = t.contiguous()
    edit_dir().mkdir(parents=True, exist_ok=True)
    save_file({k: ov[k] for k in sorted(ov)}, str(_overlay_path()))
    ch = (t.bfloat16() != o.bfloat16()).sum().item()
    st = {"name": name, "numel": t.numel(), "changed_after_bf16": ch, "frac": ch / t.numel(),
          "max_abs_change": float((t - o).abs().max())}
    if ch == 0:
        st["warning"] = "no element changes after bf16 rounding: the edit has no effect"
    return st


def edit_summary() -> dict:
    ov = _overlay()
    tot = ch = 0
    per = {}
    for k, t in ov.items():
        o = orig_tensor(k)
        c = (t.bfloat16() != o.bfloat16()).sum().item()
        per[k] = {"changed": c, "numel": t.numel(), "max_abs": float((t.bfloat16().float() - o.bfloat16().float()).abs().max())}
        ch += c
    n_all = sum(math.prod(_orig().get_slice(k).get_shape()) for k in tensor_names())
    return {"tensors": len(ov), "changed_weights": ch, "total_weights": n_all, "pct_changed": 100 * ch / n_all, "per_tensor": per}


# --------------------------------------------------------------------------- models
_tok = None
_models: dict = {}


def tokenizer():
    global _tok
    if _tok is None:
        from transformers import AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    return _tok


def _load(attn: str = "sdpa"):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR), dtype=torch.bfloat16, attn_implementation=attn,
                                             local_files_only=True)
    return m.to(DEVICE).eval()


def get_model(which: str, attn: str = "sdpa"):
    key = (which, attn, edit_hash() if which == "edit" else "orig")
    if key not in _models:
        for k in [k for k in _models if k[0] == which and k[1] == attn]:
            del _models[k]
        m = _load(attn)
        if which == "edit":
            sd = m.state_dict()
            for name, t in _overlay().items():
                if name not in sd:
                    raise HarnessError(f"tensor {name} not in model state")
                sd[name].copy_(t.to(torch.bfloat16))
        _models[key] = m
    return _models[key]


# --------------------------------------------------------------------------- language id (heuristic)
_W = {
    "sv": "och att är inte jag ett av från också någon mycket när där hade blev sig vi det som för på med den har var till eller kan en om så men de man efter under över sin sina vid ut upp mer än hur vad vem vilken vilka".split(),
    "en": "the and is of to in that it was for with as on are this be by from or an at not have had which but they his her their been has were will would can what who how when where".split(),
    "da": "og at er ikke jeg af fra også nogen meget når hvor havde blev sig vi det som for på med den har var til eller kan en om så men de man efter under over sin sine ved ud op mere end hvordan hvad hvem".split(),
    "no": "og å er ikke jeg av fra også noen mye når hvor hadde ble seg vi det som for på med den har var til eller kan en om så men de man etter under over sin sine ved ut opp mer enn hvordan hva hvem".split(),
}
_UNIQ = {l: set(w) - set().union(*[set(v) for k, v in _W.items() if k != l]) for l, w in _W.items()}
_WORD = re.compile(r"[a-zåäöæøéü]+", re.I)


def langid(text: str) -> str:
    """Rough language label: sv / en / da / no / other (function-word voting; heuristic, not a classifier)."""
    ws = [w.lower() for w in _WORD.findall(text)]
    if len(ws) < 3:
        return "other"
    sc = {}
    for l in _W:
        sc[l] = sum(1 for w in ws if w in _UNIQ[l]) * 2 + sum(1 for w in ws if w in _W[l] and w not in _UNIQ[l])
    # Swedish-only letters / Danish-Norwegian letters as tie-breakers
    sc["sv"] += 0.5 * len(re.findall(r"[äö]", text.lower())); sc["da"] += 0.5 * len(re.findall(r"[æø]", text.lower()))
    sc["no"] += 0.5 * len(re.findall(r"[æø]", text.lower()))
    best = max(sc, key=sc.get)
    return best if sc[best] >= 2 else "other"


# --------------------------------------------------------------------------- scoring primitives
def _chat(prompt: str) -> str:
    return tokenizer().apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)


def _ids(s: str) -> list[int]:
    return tokenizer()(s, add_special_tokens=False).input_ids


def _logits_lp(model, hidden: torch.Tensor, positions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """log p(target | position) from final hidden states at the given positions."""
    h = hidden[positions]
    logits = model.lm_head(h).float()
    cap = getattr(model.config, "final_logit_softcapping", None)
    if cap:
        logits = torch.tanh(logits / cap) * cap
    lp = torch.log_softmax(logits, -1)
    return lp.gather(-1, targets[:, None])[:, 0]


@torch.no_grad()
def _teacher_forced(model, seqs: list[tuple[list[int], int]], bs_tokens: int = 6000) -> list[torch.Tensor]:
    """seqs: (token ids, first scored index). Returns per-seq tensor of log p of tokens[start:]."""
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i][0]))
    out: list = [None] * len(seqs)
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and (j - i + 1) * len(seqs[order[j]][0]) <= bs_tokens:
            j += 1
        j = max(j, i + 1)
        idx = order[i:j]
        L = max(len(seqs[k][0]) for k in idx)
        ids = torch.zeros(len(idx), L, dtype=torch.long)
        mask = torch.zeros(len(idx), L, dtype=torch.long)
        for r, k in enumerate(idx):
            s = seqs[k][0]
            ids[r, : len(s)] = torch.tensor(s)
            mask[r, : len(s)] = 1
        h = model.model(input_ids=ids.to(DEVICE), attention_mask=mask.to(DEVICE)).last_hidden_state
        for r, k in enumerate(idx):
            s, st = seqs[k]
            pos = torch.arange(st - 1, len(s) - 1, device=DEVICE)
            tgt = torch.tensor(s[st:], device=DEVICE)
            out[k] = _logits_lp(model, h[r], pos, tgt).cpu()
        i = j
    return out


def _score_items(model, items: list[dict]) -> list[dict]:
    tok = tokenizer()
    seqs, owners = [], []
    for n, it in enumerate(items):
        if it["kind"] == "mc":
            p = _ids(_chat(it["prompt"]))
            for c, ch in enumerate(it["choices"]):
                a = _ids(ch + "<end_of_turn>")
                seqs.append((p + a, len(p))); owners.append((n, c))
        elif it["kind"] == "lm":
            ids = [tok.bos_token_id] + _ids(it["text"])
            seqs.append((ids, 1)); owners.append((n, None))
        else:
            raise HarnessError(f"score needs kind mc or lm, got {it['kind']}")
    lps = _teacher_forced(model, seqs)
    res: list[dict] = [dict() for _ in items]
    for (n, c), lp in zip(owners, lps):
        it = items[n]
        if it["kind"] == "mc":
            res[n].setdefault("choice_logp", {})[c] = float(lp.sum())
        else:
            res[n] = {"nll": float(-lp.sum()), "n_tokens": int(lp.numel()), "n_chars": len(it["text"])}
    for n, it in enumerate(items):
        if it["kind"] == "mc":
            d = res[n].pop("choice_logp")
            lps_ = [d[c] for c in range(len(it["choices"]))]
            res[n] = {"choice_logp": lps_, "pred": int(np.argmax(lps_))}
            if "gold" in it:
                m = max(lps_); lse = m + math.log(sum(math.exp(x - m) for x in lps_))
                res[n]["correct"] = bool(res[n]["pred"] == it["gold"])
                res[n]["choice_nll"] = float(lse - lps_[it["gold"]])
    return res


@torch.no_grad()
def _generate_items(model, items: list[dict], max_new: int, bs: int = 64) -> list[dict]:
    tok = tokenizer()
    tok.padding_side = "left"
    order = sorted(range(len(items)), key=lambda i: len(items[i]["prompt"]))
    out: list = [None] * len(items)
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]
        texts = [_chat(items[k]["prompt"]) for k in idx]
        enc = tok(texts, add_special_tokens=False, return_tensors="pt", padding=True).to(DEVICE)
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None, top_k=None)
        for r, k in enumerate(idx):
            t = tok.decode(gen[r, enc.input_ids.shape[1]:], skip_special_tokens=True).strip()
            out[k] = {"text": t, "lang": langid(t)}
    return out


def _rep(text: str) -> bool:
    w = text.lower().split()
    tri = Counter(zip(w, w[1:], w[2:]))
    return bool(tri) and max(tri.values()) >= 3


def summarize(items: list[dict], res: list[dict]) -> dict:
    by: dict[str, list] = {}
    for it, r in zip(items, res):
        by.setdefault(it["task"], []).append((it, r))
    out = {}
    for t, rows in by.items():
        k = rows[0][0]["kind"]
        n = len(rows)
        if k == "mc":
            out[t] = {"n": n, "acc": sum(r["correct"] for _, r in rows) / n,
                      "choice_nll": sum(r["choice_nll"] for _, r in rows) / n}
        elif k == "lm":
            nll = sum(r["nll"] for _, r in rows); nt = sum(r["n_tokens"] for _, r in rows); nc = sum(r["n_chars"] for _, r in rows)
            out[t] = {"n": n, "bpc": nll / math.log(2) / nc, "nll_per_token": nll / nt, "n_tokens": nt}
        else:
            c = Counter(r["lang"] for _, r in rows)
            out[t] = {"n": n, "sv_rate": c["sv"] / n, "en_rate": c["en"] / n, "da_no_rate": (c["da"] + c["no"]) / n,
                      "other_rate": c["other"] / n, "empty_rate": sum(1 for _, r in rows if not r["text"]) / n,
                      "repeat_rate": sum(_rep(r["text"]) for _, r in rows) / n,
                      "mean_chars": sum(len(r["text"]) for _, r in rows) / n}
    return out


# --------------------------------------------------------------------------- probes (attention / hidden / mlp)
@torch.no_grad()
def _probe(model, items: list[dict], what: str, layers: list[int], positions) -> dict:
    tok = tokenizer()
    res = {"tokens": [], what: {l: [] for l in layers}}
    hooks, store = [], {}
    if what == "mlp":
        for l in layers:
            hooks.append(model.model.layers[l].mlp.down_proj.register_forward_pre_hook(
                lambda m, a, l=l: store.__setitem__(l, a[0].detach())))
    try:
        for it in items:
            text = _chat(it["prompt"]) if "prompt" in it else it["text"]
            ids = torch.tensor([_ids(text) if "prompt" in it else [tok.bos_token_id] + _ids(text)], device=DEVICE)
            o = model.model(input_ids=ids, output_attentions=(what == "attn"), output_hidden_states=(what == "hidden"))
            T = ids.shape[1]
            pos = [T - 1] if positions == "last" else (list(range(T)) if positions == "all" else [p % T for p in positions])
            res["tokens"].append(tok.convert_ids_to_tokens(ids[0]))
            for l in layers:
                if what == "attn":
                    res[what][l].append(o.attentions[l][0].float().cpu().numpy())
                elif what == "hidden":
                    res[what][l].append(o.hidden_states[l + 1][0, pos].float().cpu().numpy())
                else:
                    res[what][l].append(store[l][0, pos].float().cpu().numpy())
    finally:
        for h in hooks:
            h.remove()
    return res


# --------------------------------------------------------------------------- the counted forward pass
def _request_hash(req: dict, which: str) -> str:
    h = hashlib.sha256(json.dumps(req, sort_keys=True, ensure_ascii=False).encode())
    h.update(which.encode())
    if which in ("edit", "both"):
        h.update(edit_hash().encode())
    return h.hexdigest()[:20]


def fwd(kind: str, items: list[dict], which: str = "both", *, max_new: int | None = None,
        layers: list[int] | None = None, what: str | None = None, positions="last", note: str = "") -> dict:
    """One counted pass.

    kind="score"    items of kind mc / lm. Teacher-forced log-probabilities; returns per-item results and a
                    per-task summary (acc, choice_nll, bpc, ...). Up to max_items_per_pass items.
    kind="generate" items of kind gen ({"prompt"}), greedy, up to 48 new tokens. Returns text and language label.
    kind="probe"    what in {"attn","hidden","mlp"}, layers=[...], positions="last"|"all"|[idx,...]; items are
                    {"prompt"} (chat template) or {"text"} (raw). attn: <=16 items; hidden/mlp: <=1024 items.
                    Returns numpy arrays (not stored in the log).
    which           "orig", "edit" or "both" (the same items through both models, one pass).

    An identical request with an unchanged edit is answered from cache and is free.
    A request that fails validation or crashes is not counted."""
    c = config()
    if which not in ("orig", "edit", "both"):
        raise HarnessError("which must be orig, edit or both")
    if kind not in ("score", "generate", "probe"):
        raise HarnessError("kind must be score, generate or probe")
    cap = c["max_items_per_pass"]
    if kind == "probe" and what == "attn":
        cap = 16
    if not items or len(items) > cap:
        raise HarnessError(f"need 1..{cap} items, got {len(items)}")
    max_new = min(max_new or c["max_new_tokens"], c["max_new_tokens"])
    if kind == "score" and any(i["kind"] not in ("mc", "lm") for i in items):
        raise HarnessError("score takes kind mc or lm items")
    if kind == "generate" and any("prompt" not in i for i in items):
        raise HarnessError("generate takes items with a prompt")
    if kind == "probe":
        if what not in ("attn", "hidden", "mlp") or not layers:
            raise HarnessError("probe needs what and layers")
        if any(not 0 <= l < 18 for l in layers):
            raise HarnessError("layers must be in 0..17")
    req = {"kind": kind, "items": items, "max_new": max_new, "what": what, "layers": layers, "positions": positions}
    key = _request_hash(req, which)
    rd = run_dir()
    cache = rd / "cache" / f"{key}.pkl"
    if cache.exists():
        r = pickle.load(cache.open("rb")); r["cached"] = True
        print(f"[fwd] identical request: served from cache, not counted ({status_line()})")
        return r

    lock = (rd / ".lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    try:
        used = passes_used()
        if used >= c["budget"]:
            raise BudgetExhausted(f"all {c['budget']} passes are used")
        t0 = time.time()
        sides = ["orig", "edit"] if which == "both" else [which]
        r: dict = {"pass": used + 1, "kind": kind, "which": which}
        for s in sides:
            m = get_model(s, "eager" if (kind == "probe" and what == "attn") else "sdpa")
            if kind == "score":
                per = _score_items(m, items)
                r[s] = {"per_item": per, "summary": summarize(items, per)}
            elif kind == "generate":
                per = _generate_items(m, items, max_new)
                r[s] = {"per_item": per, "summary": summarize(items, per)}
            else:
                r[s] = _probe(m, items, what, layers, positions)
        if which == "both" and kind != "probe":
            r["delta"] = _delta(r["orig"]["summary"], r["edit"]["summary"])
        r["remaining"] = c["budget"] - (used + 1)
        r["cached"] = False
        rec = {"type": "pass", "n": used + 1, "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "kind": kind, "which": which, "n_items": len(items), "tasks": dict(Counter(i.get("task", "?") for i in items)),
               "edit_hash": edit_hash(), "req_hash": key, "secs": round(time.time() - t0, 1), "note": note}
        with _log_path().open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        cache.parent.mkdir(exist_ok=True)
        pickle.dump(r, cache.open("wb"))
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN); lock.close()
    print(f"[fwd] pass {r['pass']}/{c['budget']} done in {rec['secs']}s; remaining {r['remaining']}")
    return r


def status_line() -> str:
    c = config()
    return f"{passes_used()}/{c['budget']} passes used"


def _delta(a: dict, b: dict) -> dict:
    d = {}
    for t in a:
        d[t] = {k: b[t][k] - a[t][k] for k in a[t] if isinstance(a[t][k], float)}
    return d


def show(res: dict) -> None:
    """Compact table of a fwd() result."""
    if res["kind"] == "probe":
        print("probe result; arrays under res['orig'|'edit'][what][layer]")
        return
    sides = [s for s in ("orig", "edit") if s in res]
    tasks = list(res[sides[0]]["summary"])
    for t in tasks:
        keys = [k for k in res[sides[0]]["summary"][t] if k not in ("n", "n_tokens")]
        n = res[sides[0]]["summary"][t]["n"]
        print(f"{t} (n={n})")
        for k in keys:
            vals = "  ".join(f"{s}={res[s]['summary'][t][k]:.4f}" for s in sides)
            dl = f"  delta={res['delta'][t][k]:+.4f}" if "delta" in res else ""
            print(f"   {k:14s} {vals}{dl}")


# --------------------------------------------------------------------------- helpers reused by final_eval.py
def load_jsonl(path: Path, n: int | None = None) -> list[dict]:
    rows = [json.loads(ln) for ln in Path(path).read_text().splitlines() if ln.strip()]
    return rows if n is None else rows[:n]


def model_from_dir(path: str):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                local_files_only=True).to(DEVICE).eval()
