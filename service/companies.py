"""Company mention detection for metadata-filtered retrieval.

Nearest-neighbour retrieval over 50 filings often returns another company's chunk (the original
study retrieved the right filing for only 39 of 68 FinanceBench questions). When the question
names an indexed company, retrieval is restricted to that company's chunks with a SQL filter
(`WHERE company = ANY(...)`), so the false-premise check compares the premise with the right
filing. Questions naming no indexed company are searched unfiltered.

Values are the `company` strings stored in the chunks table (two filings of Microsoft and Amazon
carry different spellings, so an alias can map to several values).
"""
from __future__ import annotations

import re

# alias regex (case-insensitive unless noted) -> company values in the corpus
_ALIASES: list[tuple[str, tuple[str, ...], int]] = [
    (r"\bjp\s?morgan\b|\bjpmorgan\b|\bjpm\b", ("JPMorgan",), re.I),
    (r"\bAES\b", ("AES Corporation",), 0),
    (r"\bCVS\b", ("CVS Health",), 0),
    (r"\bcoca[- ]cola\b", ("Coca-Cola",), re.I),
    (r"\bpfizer\b", ("Pfizer",), re.I),
    (r"\bamerican water\b", ("American Water Works",), re.I),
    (r"\bamerican express\b|\bamex\b", ("American Express",), re.I),
    (r"\b3M\b", ("3M",), 0),
    (r"\bMeta\b|\b[Ff]acebook\b|\b[Ii]nstagram\b|\b[Ww]hats[Aa]pp\b", ("Meta Platforms Inc.",), 0),
    (r"\blockheed\b", ("Lockheed Martin",), re.I),
    (r"\bkraft\b|\bheinz\b", ("Kraft Heinz",), re.I),
    (r"\bjohnson\s*(&|and)\s*johnson\b|\bJ&J\b|\bjnj\b", ("Johnson & Johnson",), re.I),
    (r"\bactivision\b|\bblizzard\b", ("Activision Blizzard",), re.I),
    (r"\bpepsico\b|\bpepsi\b", ("PepsiCo",), re.I),
    (r"\btesla\b", ("Tesla Inc.",), re.I),
    (r"\bcorning\b", ("Corning",), re.I),
    (r"\bnike\b", ("Nike",), re.I),
    (r"\bamcor\b", ("Amcor",), re.I),
    (r"\bwalmart\b|\bwal-mart\b", ("Walmart",), re.I),
    (r"\balphabet\b|\bgoogle\b|\byoutube\b", ("Alphabet Inc.",), re.I),
    (r"\bnvidia\b", ("NVIDIA Corp.",), re.I),
    (r"\bbest buy\b", ("Best Buy",), re.I),
    (r"\bmicrosoft\b", ("Microsoft", "Microsoft Corp."), re.I),
    (r"\bamazon\b|\bAWS\b", ("Amazon", "Amazon.com Inc."), re.I),
    (r"\bulta\b", ("Ulta Beauty",), re.I),
    (r"\bcostco\b", ("Costco",), re.I),
    (r"\bapple\b|\biphone\b|\bipad\b|\bmac\b", ("Apple Inc.",), re.I),
]
_COMPILED = [(re.compile(p, f), v) for p, v, f in _ALIASES]


def detect_companies(question: str) -> list[str]:
    """Indexed company values mentioned in the question (empty list = no filter)."""
    out: list[str] = []
    for rx, values in _COMPILED:
        if rx.search(question):
            out.extend(v for v in values if v not in out)
    return out
