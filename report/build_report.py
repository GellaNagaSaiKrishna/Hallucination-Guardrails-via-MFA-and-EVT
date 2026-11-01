"""Render report/final_report.md to report/final_report.pdf with ReportLab.

Supports: # / ## / ### headings, paragraphs, - bullets, 1. numbered items, | tables |,
![caption](path) images, ``` code blocks ```, $$ display equations $$ (rendered as monospace),
**bold**, *italic*, `code`, [links](url).

Usage:  python report/build_report.py
"""
from __future__ import annotations

import re
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (HRFlowable, Image, KeepTogether, PageBreak, Paragraph, Preformatted,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PRIMARY = colors.HexColor("#1E3A8A")
INK = colors.HexColor("#1F2937")
MUTED = colors.HexColor("#52514e")
BORDER = colors.HexColor("#D1D5DB")


def _register_fonts():
    """Use DejaVu/Arial (Unicode: Greek, ², ≤, →) when available, else Helvetica."""
    candidates = [
        ("Body", ["C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]),
        ("Body-Bold", ["C:/Windows/Fonts/arialbd.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]),
        ("Body-Italic", ["C:/Windows/Fonts/ariali.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf"]),
        ("Mono", ["C:/Windows/Fonts/consola.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"]),
    ]
    found = {}
    for name, paths in candidates:
        for p in paths:
            if Path(p).exists():
                pdfmetrics.registerFont(TTFont(name, p))
                found[name] = name
                break
    if len(found) == 4:
        from reportlab.pdfbase.pdfmetrics import registerFontFamily
        registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body-Italic", boldItalic="Body-Bold")
        return "Body", "Body-Bold", "Mono"
    return "Helvetica", "Helvetica-Bold", "Courier"


BODY, BOLD, MONO = _register_fonts()
ss = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("t", parent=ss["Normal"], fontName=BOLD, fontSize=17, leading=21, textColor=PRIMARY,
                            spaceAfter=4),
    "subtitle": ParagraphStyle("st", parent=ss["Normal"], fontName=BODY, fontSize=9.5, leading=13, textColor=MUTED,
                               spaceAfter=6),
    "h2": ParagraphStyle("h2", parent=ss["Normal"], fontName=BOLD, fontSize=13, leading=16, textColor=PRIMARY,
                         spaceBefore=10, spaceAfter=4, keepWithNext=True),
    "h3": ParagraphStyle("h3", parent=ss["Normal"], fontName=BOLD, fontSize=10.5, leading=13.5, textColor=INK,
                         spaceBefore=7, spaceAfter=3, keepWithNext=True),
    "body": ParagraphStyle("b", parent=ss["Normal"], fontName=BODY, fontSize=9.3, leading=13, textColor=INK,
                           spaceAfter=4, alignment=TA_JUSTIFY),
    "bullet": ParagraphStyle("bl", parent=ss["Normal"], fontName=BODY, fontSize=9.3, leading=12.6, textColor=INK,
                             leftIndent=14, bulletIndent=4, spaceAfter=2),
    "caption": ParagraphStyle("c", parent=ss["Normal"], fontName=BODY, fontSize=8.2, leading=10.5, textColor=MUTED,
                              alignment=TA_CENTER, spaceAfter=8),
    "th": ParagraphStyle("th", parent=ss["Normal"], fontName=BOLD, fontSize=7.8, leading=9.6, textColor=colors.white),
    "td": ParagraphStyle("td", parent=ss["Normal"], fontName=BODY, fontSize=7.8, leading=9.6, textColor=INK),
    "code": ParagraphStyle("code", parent=ss["Normal"], fontName=MONO, fontSize=8, leading=10.5, textColor=INK,
                           backColor=colors.HexColor("#F3F4F6"), borderPadding=5, leftIndent=6, spaceAfter=6,
                           spaceBefore=2),
}


def inline(t: str) -> str:
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    t = re.sub(r"`([^`]+)`", rf'<font name="{MONO}">\1</font>', t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", t)
    t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<link href="\2" color="#2563EB"><u>\1</u></link>', t)
    return t


def make_table(rows: list[list[str]], width: float):
    ncol = len(rows[0])
    data = [[Paragraph(inline(c), S["th"]) for c in rows[0]]]
    data += [[Paragraph(inline(c), S["td"]) for c in r + [""] * (ncol - len(r))] for r in rows[1:]]
    # first column wider for text-heavy tables
    lens = [max(len(r[i]) if i < len(r) else 0 for r in rows) for i in range(ncol)]
    tot = sum(min(l, 60) for l in lens) or 1
    widths = [width * max(min(l, 60) / tot, 0.07) for l in lens]
    scale = width / sum(widths)
    t = Table(data, colWidths=[w * scale for w in widths], repeatRows=1, hAlign="CENTER")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PRIMARY), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, BORDER), ("BOX", (0, 0), (-1, -1), 0.6, PRIMARY),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9FAFB")]),
    ]))
    return t


def make_image(path: Path, caption: str, width: float):
    with PILImage.open(path) as im:
        w, h = im.size
    max_h = 4.3 * inch
    tw = width
    th = tw * h / w
    if th > max_h:
        th, tw = max_h, max_h * w / h
    return KeepTogether([Image(str(path), width=tw, height=th), Paragraph(inline(caption), S["caption"])])


def build(md_path: Path, pdf_path: Path):
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, leftMargin=0.7 * inch, rightMargin=0.7 * inch,
                            topMargin=0.6 * inch, bottomMargin=0.6 * inch, title="Final Report",
                            author="Gella Naga Sai Krishna")
    W = doc.width
    lines = md_path.read_text(encoding="utf-8").splitlines()
    flow, para, table, i = [], [], [], 0
    first_heading = True

    def flush_para():
        if para:
            flow.append(Paragraph(inline(" ".join(para)), S["body"]))
            para.clear()

    def flush_table():
        if table:
            flow.append(make_table(table, W))
            flow.append(Spacer(1, 6))
            table.clear()

    def page_num(canvas, _doc):
        canvas.saveState()
        canvas.setFont(BODY, 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawRightString(A4[0] - 0.7 * inch, 0.35 * inch, f"{_doc.page}")
        canvas.restoreState()

    while i < len(lines):
        raw = lines[i]
        s = raw.strip()
        if s.startswith("```") or s == "$$":
            flush_para(); flush_table()
            end = "```" if s.startswith("```") else "$$"
            buf = []
            i += 1
            while i < len(lines) and lines[i].strip() != end and not lines[i].strip().startswith(end if end == "$$" else "```"):
                buf.append(lines[i])
                i += 1
            flow.append(Preformatted("\n".join(buf), S["code"]))
            i += 1
            continue
        if s.startswith("|") and s.endswith("|"):
            flush_para()
            if not re.fullmatch(r"\|[\s\-:|]+\|", s):
                table.append([c.strip() for c in s.strip("|").split("|")])
            i += 1
            continue
        flush_table()
        if not s:
            flush_para()
        elif s == "<!-- pagebreak -->":
            flush_para(); flow.append(PageBreak())
        elif s.startswith("# "):
            flush_para()
            flow.append(Paragraph(inline(s[2:]), S["title"]))
        elif s.startswith("## "):
            flush_para()
            flow.append(Paragraph(inline(s[3:]), S["h2"]))
            flow.append(HRFlowable(width="100%", thickness=0.5, color=BORDER, spaceAfter=3))
        elif s.startswith("### "):
            flush_para()
            flow.append(Paragraph(inline(s[4:]), S["h3"]))
        elif s.startswith("> "):
            flush_para()
            flow.append(Paragraph(inline(s[2:]), S["subtitle"]))
        elif m := re.match(r"!\[(.*)\]\((.+)\)", s):
            flush_para()
            flow.append(make_image((md_path.parent / m.group(2)).resolve(), m.group(1), W))
        elif s.startswith(("- ", "* ")):
            flush_para()
            flow.append(Paragraph(inline(s[2:]), S["bullet"], bulletText="•"))
        elif m := re.match(r"(\d+)\.\s+(.*)", s):
            flush_para()
            flow.append(Paragraph(inline(m.group(2)), S["bullet"], bulletText=f"{m.group(1)}."))
        elif s == "---":
            flush_para()
            flow.append(HRFlowable(width="100%", thickness=0.5, color=BORDER, spaceBefore=3, spaceAfter=3))
        else:
            para.append(s)
        i += 1
    flush_para(); flush_table()
    doc.build(flow, onFirstPage=page_num, onLaterPages=page_num)
    print("wrote", pdf_path)


if __name__ == "__main__":
    build(HERE / "final_report.md", HERE / "final_report.pdf")
