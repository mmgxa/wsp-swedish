"""Shared item lists for my passes (written by me, not from the holdout)."""
import sys
sys.path.insert(0, "sv")
import harness as H

# Hand-written open-ended Swedish prompts (diagnosis only; not used for scoring decisions alone).
SV_PROMPTS = [
    "Skriv en kort berättelse om en katt som bor i Stockholm.",
    "Förklara hur fotosyntes fungerar, med enkla ord.",
    "Vad är skillnaden mellan en sjö och ett hav?",
    "Ge mig tre tips för att sova bättre.",
    "Beskriv en typisk svensk midsommarfest.",
    "Varför är himlen blå?",
    "Skriv ett kort mejl till min chef om att jag är sjuk i dag.",
    "Vad betyder ordet 'lagom'?",
    "Hur lagar man pannkakor? Svara kort.",
    "Berätta om Sveriges huvudstad.",
    "Vilka är fördelarna med att cykla till jobbet?",
    "Sammanfatta handlingen i Romeo och Julia i två meningar.",
    "Vad är en vulkan?",
    "Ge ett exempel på en mening med ordet 'eftersom'.",
    "Hur många dagar har ett skottår?",
    "Vad gör en bibliotekarie?",
    "Skriv en dikt om hösten.",
    "Förklara vad en dator är för ett barn.",
    "Vilket är världens största djur?",
    "Vad är det bästa sättet att lära sig ett nytt språk?",
    "Vem skrev Pippi Långstrump?",
    "Vad heter Norges huvudstad?",
    "Hur fungerar ett kylskåp?",
    "Varför ska man äta grönsaker?",
    "Beskriv vädret i Sverige på vintern.",
    "Vad är skillnaden mellan svenska och norska?",
    "Ge mig ett recept på köttbullar.",
    "Vad är klimatförändringar?",
    "Hur säger man 'tack' på engelska?",
    "Vem är kung i Sverige?",
    "Skriv en mening om en hund som leker i snön.",
    "Vilka färger har den svenska flaggan?",
    "Vad är ett primtal?",
    "Berätta en rolig historia.",
]


def gen_items():
    dev = H.dev_items("gen", 30)
    own = [{"id": f"own-{i}", "task": "gen_own", "kind": "gen", "prompt": p} for i, p in enumerate(SV_PROMPTS)]
    return dev + own
