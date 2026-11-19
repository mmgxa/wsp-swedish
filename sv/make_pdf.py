"""Render RESEARCH_NOTE.md to PDF with reportlab.

    uv run --offline python sv/make_pdf.py [--md RESEARCH_NOTE.md] [--out RESEARCH_NOTE.pdf]

Handles the Markdown subset used in the note: #/## headings, paragraphs, bullets, numbered lists (one nesting level),
pipe tables, one image, **bold**, *italic*, `code`, [links](url).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, KeepTogether, ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parent.parent
FONTS = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
for name, f in [("DV", "DejaVuSans.ttf"), ("DV-B", "DejaVuSans-Bold.ttf"), ("DV-I", "DejaVuSans-Oblique.ttf"),
                ("DV-BI", "DejaVuSans-BoldOblique.ttf"), ("DVM", "DejaVuSansMono.ttf")]:
    pdfmetrics.registerFont(TTFont(name, str(FONTS / f)))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")

INK, BLUE, GREY = colors.HexColor("#1f2933"), colors.HexColor("#1f4e79"), colors.HexColor("#d5dbe1")
BASE = 9.0
S = {
    "title": ParagraphStyle("title", fontName="DV-B", fontSize=14, leading=17.5, textColor=BLUE, spaceAfter=4),
    "h2": ParagraphStyle("h2", fontName="DV-B", fontSize=10.5, leading=13, textColor=BLUE, spaceBefore=6, spaceAfter=2.5, keepWithNext=1),
    "p": ParagraphStyle("p", fontName="DV", fontSize=BASE, leading=BASE * 1.38, textColor=INK, spaceAfter=3, alignment=TA_LEFT),
    "li": ParagraphStyle("li", fontName="DV", fontSize=BASE, leading=BASE * 1.38, textColor=INK, spaceAfter=1.2),
    "cell": ParagraphStyle("cell", fontName="DV", fontSize=BASE - 1.2, leading=(BASE - 1.2) * 1.3, textColor=INK),
    "head": ParagraphStyle("head", fontName="DV-B", fontSize=BASE - 1.2, leading=(BASE - 1.2) * 1.3, textColor=colors.white),
}


def inline(t: str) -> str:
    parts = re.split(r"(`[^`]+`)", t)
    out = []
    for p in parts:
        if p.startswith("`") and p.endswith("`") and len(p) > 1:
            out.append(f'<font name="DVM" size="{BASE - 0.6}" color="#8a2d3b">{escape(p[1:-1])}</font>')
            continue
        p = escape(p)
        p = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r'<a href="\2" color="#1f4e79"><u>\1</u></a>', p)
        p = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", p)
        p = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", p)
        out.append(p)
    return "".join(out)


def table(rows: list[list[str]], width: float):
    n = len(rows[0])
    wts = [max(8, min(34, max(len(re.sub(r"[`*]", "", r[i])) for r in rows))) for i in range(n)]
    wts = [w ** .85 for w in wts]
    cw = [width * w / sum(wts) for w in wts]
    data = [[Paragraph(inline(c).replace("#8a2d3b", "#ffffff") if r == 0 else inline(c), S["head"] if r == 0 else S["cell"]) for c in row]
            for r, row in enumerate(rows)]
    t = Table(data, colWidths=cw, repeatRows=1)
    st = [("BACKGROUND", (0, 0), (-1, 0), BLUE), ("GRID", (0, 0), (-1, -1), .4, GREY), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
          ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]
    for r in range(2, len(rows), 2):
        st.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#f2f5f8")))
    t.setStyle(TableStyle(st))
    return t


def build(md_path: Path, out: Path, fig_w_cm: float):
    lines = md_path.read_text().splitlines()
    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=1.7 * cm, rightMargin=1.7 * cm, topMargin=1.4 * cm,
                            bottomMargin=1.5 * cm, title="Research note: weight-space programming for Swedish on Gemma-3-270M-it",
                            author="")
    width = A4[0] - 3.4 * cm
    flow, i = [], 0
    while i < len(lines):
        ln = lines[i]
        if not ln.strip():
            i += 1
        elif ln.startswith("# "):
            flow.append(Paragraph(inline(ln[2:]), S["title"])); i += 1
        elif ln.startswith("## "):
            flow.append(Paragraph(inline(ln[3:]), S["h2"])); i += 1
        elif ln.startswith("!["):
            path = ROOT / re.match(r"!\[.*?\]\((.+?)\)", ln).group(1)
            from PIL import Image as PI
            w, h = PI.open(path).size
            flow += [Image(str(path), width=fig_w_cm * cm, height=fig_w_cm * cm * h / w, hAlign="CENTER"), Spacer(1, 3)]; i += 1
        elif ln.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            flow += [table(rows, width), Spacer(1, 4)]
        elif re.match(r"^(- |\d+\. )", ln):
            items, ordered = [], bool(re.match(r"^\d+\. ", ln))
            while i < len(lines) and re.match(r"^(- |\d+\. )", lines[i]):
                txt = re.sub(r"^(- |\d+\. )", "", lines[i]); i += 1
                sub = []
                while i < len(lines) and re.match(r"^\s{2,}(\d+\. |- )", lines[i]):
                    sub.append(Paragraph(inline(re.sub(r"^\s+(\d+\. |- )", "", lines[i])), S["li"])); i += 1
                content = [Paragraph(inline(txt), S["li"])]
                if sub:
                    content.append(ListFlowable([ListItem(x) for x in sub], bulletType="1", bulletFontName="DV", bulletFontSize=BASE,
                                                leftIndent=14, bulletFormat="%s."))
                items.append(ListItem(content))
            flow.append(ListFlowable(items, bulletType="1" if ordered else "bullet", bulletFontName="DV", bulletFontSize=BASE,
                                     leftIndent=13, bulletFormat="%s." if ordered else None, start=1 if ordered else "•"))
            flow.append(Spacer(1, 2))
        else:
            para = [ln.strip()]; i += 1
            while i < len(lines) and lines[i].strip() and not re.match(r"^(#|\||!\[|- |\d+\. )", lines[i]):
                para.append(lines[i].strip()); i += 1
            flow.append(Paragraph(inline(" ".join(para)), S["p"]))

    def footer(c, d):
        c.saveState(); c.setFont("DV", 7); c.setFillColor(colors.HexColor("#6b7785"))
        c.drawRightString(A4[0] - 1.7 * cm, .8 * cm, f"{d.page}"); c.restoreState()

    doc.build(flow, onFirstPage=footer, onLaterPages=footer)
    return doc.page


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default=str(ROOT / "RESEARCH_NOTE.md"))
    ap.add_argument("--out", default=str(ROOT / "RESEARCH_NOTE.pdf"))
    ap.add_argument("--fig-width-cm", type=float, default=15.0)
    a = ap.parse_args()
    n = build(Path(a.md), Path(a.out), a.fig_width_cm)
    print(f"{a.out}: {n} page(s)")
