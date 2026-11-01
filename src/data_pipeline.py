"""Week 1 / Step 1.1 - Document ingestion and chunking.

Downloads (if missing) the SEC 10-K filings, extracts readable text from the
inline-XBRL HTML, strips boilerplate (hidden XBRL header, page numbers,
repeated running headers) and splits the text with LangChain's
RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).

Usage:  python -m src.data_pipeline
"""
from __future__ import annotations

import json
import re
import time
import urllib.request
import warnings
from collections import Counter

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import CHUNK_OVERLAP, CHUNK_SIZE, CHUNKS_JSON, RAW

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

HEADERS = {"User-Agent": "AcademicResearchStudent cs328@univ.edu"}

FILINGS = [
    ("AAPL", "Apple Inc.", "2025",
     "https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm"),
    ("MSFT", "Microsoft Corp.", "2026",
     "https://www.sec.gov/Archives/edgar/data/789019/000119312526323660/msft-20260630.htm"),
    ("GOOGL", "Alphabet Inc.", "2025",
     "https://www.sec.gov/Archives/edgar/data/1652044/000165204426000018/goog-20251231.htm"),
    ("NVDA", "NVIDIA Corp.", "2026",
     "https://www.sec.gov/Archives/edgar/data/1045810/000104581026000021/nvda-20260125.htm"),
    ("AMZN", "Amazon.com Inc.", "2026",
     "https://www.sec.gov/Archives/edgar/data/1018724/000101872426000004/amzn-20251231.htm"),
    ("META", "Meta Platforms Inc.", "2026",
     "https://www.sec.gov/Archives/edgar/data/1326801/000162828026003942/meta-20251231.htm"),
    ("TSLA", "Tesla Inc.", "2026",
     "https://www.sec.gov/Archives/edgar/data/1318605/000162828026003952/tsla-20251231.htm"),
]

_BLOCK_TAGS = ["p", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"]
_NESTED_BLOCKS = ["p", "div", "li", "table", "h1", "h2", "h3", "h4", "h5", "h6"]
_WS = re.compile(r"\s+")


def download(ticker: str, url: str) -> str:
    path = RAW / f"{ticker}_10K.htm"
    if not path.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {ticker} 10-K ...")
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=60) as resp:
            path.write_text(resp.read().decode("utf-8", errors="ignore"), encoding="utf-8")
        time.sleep(1)
    return path.read_text(encoding="utf-8")


def html_to_lines(html: str) -> list[str]:
    """Return document-ordered text lines, one per leaf block / table row."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup.find_all(["ix:header", "script", "style", "head"]):
        tag.decompose()
    for tag in soup.find_all(style=re.compile(r"display:\s*none", re.I)):
        tag.decompose()

    lines: list[str] = []
    for el in soup.find_all(_BLOCK_TAGS):
        if el.name == "tr":
            cells = [_WS.sub(" ", c.get_text(" ")).strip() for c in el.find_all(["td", "th"])]
            cells = [c for c in cells if c and c not in {"$", "%", ")", "("}]
            line = " | ".join(cells)
        else:
            if el.find_parent("table") is not None or el.find(_NESTED_BLOCKS) is not None:
                continue
            line = _WS.sub(" ", el.get_text(" ")).strip()
        if line:
            lines.append(line)
    return lines


def clean_lines(lines: list[str]) -> list[str]:
    """Remove page numbers, 'Table of Contents' links and repeated running headers."""
    counts = Counter(lines)
    out = []
    for ln in lines:
        if re.fullmatch(r"[\divxlcIVXLC\-\s|]{0,8}", ln):          # page numbers / roman numerals
            continue
        if ln.lower() in {"table of contents", "index"}:
            continue
        if counts[ln] >= 5 and len(ln) < 80:                        # running headers / footers
            continue
        out.append(ln)
    return out


def is_informative(text: str) -> bool:
    words = re.findall(r"[A-Za-z]{3,}", text)
    if len(words) < 8:
        return False
    if re.search(r"https?://|fasb\.org|us-gaap:|xbrli|iso4217", text):
        return False
    alpha = sum(c.isalpha() for c in text) / max(len(text), 1)
    return alpha >= 0.35


def build_corpus() -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
        keep_separator="end",
    )
    corpus, seen = [], set()
    for ticker, company, year, url in FILINGS:
        lines = clean_lines(html_to_lines(download(ticker, url)))
        text = "\n".join(lines)
        kept = 0
        for piece in splitter.split_text(text):
            piece = piece.strip()
            if len(piece) < 100 or not is_informative(piece) or piece in seen:
                continue
            seen.add(piece)
            corpus.append({
                "chunk_id": f"{ticker}_chunk_{kept:05d}",
                "ticker": ticker,
                "company": company,
                "year": year,
                "text": piece,
                "char_count": len(piece),
            })
            kept += 1
        print(f"{ticker}: {len(text):>8,} chars -> {kept:,} chunks")
    return corpus


def main() -> None:
    corpus = build_corpus()
    CHUNKS_JSON.write_text(json.dumps(corpus, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Total chunks: {len(corpus):,}  ->  {CHUNKS_JSON}")


if __name__ == "__main__":
    main()
