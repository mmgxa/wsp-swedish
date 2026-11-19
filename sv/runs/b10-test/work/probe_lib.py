"""Item building for probe passes and analysis of returned final hidden states (no model calls)."""
import sys, math, pickle, torch, numpy as np
sys.path.insert(0, "sv"); sys.path.insert(0, "sv/runs/b10-test/work")
import harness as H, items as S

def build_items():
    its = []  # each: dict(group, req, kind, extra)
    for x in H.dev_items("lm", 30):
        its.append({"group": "lm_devA" if len(its) < 15 else "lm_devB", "req": {"text": x["text"]}, "kind": "lm", "n_chars": len(x["text"])})
    for t in S.EN_TEXTS:
        its.append({"group": "en_self", "req": {"text": t}, "kind": "lm", "n_chars": len(t)})
    for li, lang in enumerate(["sv", "en", "da", "no"]):
        for p in S.PARALLEL:
            its.append({"group": f"par_{lang}", "req": {"text": p[li]}, "kind": "lm", "n_chars": len(p[li])})
    for k, x in enumerate(H.dev_items("scala", 30)):
        chat = H._chat(x["prompt"]); assert chat.startswith("<bos>")
        p = H._ids(chat)
        for c, ch in enumerate(x["choices"]):
            full = [H.tokenizer().bos_token_id] + H._ids(chat[len("<bos>"):] + ch + "<end_of_turn>")
            a = H._ids(ch + "<end_of_turn>")
            assert full == p + a, "tokenization mismatch"
            its.append({"group": "scala", "req": {"text": chat[len("<bos>"):] + ch + "<end_of_turn>"}, "kind": "mc",
                        "q": k, "c": c, "gold": x["gold"], "start": len(p)})
    for g, prompts in [("gen_dev", [x["prompt"] for x in H.dev_items("gen", 30)]), ("gen_sv_self", S.SV_PROMPTS), ("gen_en_self", S.EN_PROMPTS)]:
        for p in prompts:
            its.append({"group": g, "req": {"prompt": p}, "kind": "prompt"})
    return its

_E = None
def logp(h):
    global _E
    if _E is None:
        _E = H.orig_tensor("model.embed_tokens.weight").to(H.DEVICE)
    h = torch.as_tensor(np.asarray(h, dtype=np.float32), device=H.DEVICE)
    return torch.log_softmax(h @ _E.T, -1)  # [T, V]

def analyse(its, tokens, hid, sides=("orig", "edit")):
    tok = H.tokenizer()
    per = {s: [] for s in sides}; kl = []
    for i, it in enumerate(its):
        ids = torch.tensor(tok.convert_tokens_to_ids(tokens[i]), device=H.DEVICE)
        lps = {s: logp(hid[s][i]) for s in sides}
        if len(sides) == 2:
            a, b = lps["orig"], lps["edit"]
            klpos = (a.exp() * (a - b)).sum(-1)
        for s in sides:
            lp = lps[s]
            tgt = lp[:-1].gather(-1, ids[1:, None])[:, 0]
            if it["kind"] == "lm":
                per[s].append(float(-tgt.sum()))
            elif it["kind"] == "mc":
                per[s].append(float(tgt[it["start"] - 1:].sum()))
            else:
                per[s].append(int(lp[-1].argmax()))
        if len(sides) == 2:
            kl.append(float(klpos[-1]) if it["kind"] == "prompt" else float(klpos[:-1].mean()))
    return per, kl

def summary(its, per, kl, sides=("orig", "edit")):
    groups = sorted({it["group"] for it in its}, key=lambda g: [it["group"] for it in its].index(g))
    out = {}
    for g in groups:
        idx = [i for i, it in enumerate(its) if it["group"] == g]
        k = its[idx[0]]["kind"]; row = {}
        for s in sides:
            if k == "lm":
                row[f"bpc_{s}"] = sum(per[s][i] for i in idx) / math.log(2) / sum(its[i]["n_chars"] for i in idx)
            elif k == "mc":
                qs = sorted({its[i]["q"] for i in idx}); acc = nll = 0
                for q in qs:
                    jj = [i for i in idx if its[i]["q"] == q]; lp = [per[s][i] for i in sorted(jj, key=lambda i: its[i]["c"])]
                    gold = its[jj[0]]["gold"]; m = max(lp); lse = m + math.log(sum(math.exp(x - m) for x in lp))
                    acc += int(np.argmax(lp) == gold); nll += lse - lp[gold]
                row[f"acc_{s}"] = acc / len(qs); row[f"cnll_{s}"] = nll / len(qs)
            else:
                row[f"top1_{s}"] = [per[s][i] for i in idx]
        if len(sides) == 2:
            row["kl"] = float(np.mean([kl[i] for i in idx]))
            if k == "prompt":
                row["top1_changed"] = sum(a != b for a, b in zip(row["top1_orig"], row["top1_edit"])) / len(idx)
        out[g] = row
    return out

def print_summary(out):
    for g, row in out.items():
        s = "  ".join(f"{k}={v:.4f}" for k, v in row.items() if isinstance(v, float))
        if "bpc_orig" in row and "bpc_edit" in row:
            s += f"  rel={100 * (row['bpc_edit'] / row['bpc_orig'] - 1):+.2f}%"
        print(f"{g:12s} {s}")
