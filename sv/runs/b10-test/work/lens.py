"""Free weight analysis: logit-lens of MLP neuron output directions on sv vs da/no vs en token pairs."""
import sys, torch, numpy as np; sys.path.insert(0, "sv"); import harness as H
tok = H.tokenizer()
def t1(w):
    ids = tok.encode(" " + w, add_special_tokens=False); assert len(ids) == 1, w; return ids[0]
SV_DANO = [("och","og"),("inte","ikke"),("jag","jeg"),("vad","hvad"),("vad","hva"),("mig","meg"),("också","også"),
           ("efter","etter"),("mycket","meget"),("mycket","mye"),("hur","hvordan"),("någon","noen"),("hade","hadde"),
           ("sig","seg"),("dig","deg"),("från","fra"),("när","når"),("bara","kun")]
SV_EN = [("och","and"),("inte","not"),("jag","I"),("är","is"),("av","of"),("vad","what"),("mig","me"),("också","also"),
         ("efter","after"),("mycket","very"),("hur","how"),("från","from"),("när","when"),("där","there"),("det","it"),
         ("att","to"),("som","which"),("med","with"),("på","on"),("var","was")]
E = H.orig_tensor("model.embed_tokens.weight")
gF = 1 + H.orig_tensor("model.norm.weight")
def pairdir(pairs):
    a = torch.stack([E[t1(s)] for s, _ in pairs]); b = torch.stack([E[t1(o)] for _, o in pairs])
    return (a - b)  # [P, 640] logit difference directions (sv minus other)
D_dano, D_en = pairdir(SV_DANO), pairdir(SV_EN)
def neuron_scores(l):
    g = 1 + H.orig_tensor(f"model.layers.{l}.post_feedforward_layernorm.weight")
    W = H.orig_tensor(f"model.layers.{l}.mlp.down_proj.weight")  # [640, 2048]
    V = (gF * g)[:, None] * W  # effective output direction (up to rms scale)
    return (D_dano @ V).mean(0), (D_en @ V).mean(0), V.norm(dim=0)
if __name__ == "__main__":
    for l in range(18):
        sd, se, nrm = neuron_scores(l)
        z = (sd - sd.mean()) / sd.std()
        top = torch.topk(-z, 3)
        print(f"L{l:2d} dano: min z {(-top.values).tolist()} idx {top.indices.tolist()} | std sd {sd.std():.3f} se {se.std():.3f}")
