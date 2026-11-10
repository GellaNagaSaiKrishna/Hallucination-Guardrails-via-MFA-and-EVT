"""Download, from SEC EDGAR, every 10-K that the open FinanceBench questions refer to.

FinanceBench (PatronusAI/financebench, CC-BY-NC-4.0) links each question to a filing such as
"3M_2018_10K".  We locate the same filing on EDGAR: the company's 10-K whose fiscal period end
(reportDate) falls in the labelled year.  Output: data/raw/fb/<doc_name>.htm + data/financebench/filings.json

Usage (from the project root):  python scripts/fetch_financebench_filings.py
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data_pipeline import HEADERS  # noqa: E402

FB = ROOT / "data" / "financebench" / "financebench_merged.jsonl"
OUT = ROOT / "data" / "raw" / "fb"
CIK = {
    "3M": 66740, "AES Corporation": 874761, "AMD": 2488, "Activision Blizzard": 718877, "Adobe": 796343,
    "Amazon": 1018724, "Amcor": 1748790, "American Express": 4962, "American Water Works": 1410636,
    "Best Buy": 764478, "Block": 1512673, "Boeing": 12927, "CVS Health": 64803, "Coca-Cola": 21344,
    "Corning": 24741, "Costco": 909832, "Foot Locker": 850209, "General Mills": 40704, "JPMorgan": 19617,
    "Johnson & Johnson": 200406, "Kraft Heinz": 1637459, "Lockheed Martin": 936468, "MGM Resorts": 789570,
    "Microsoft": 789019, "Netflix": 1065280, "Nike": 320187, "Paypal": 1633917, "PepsiCo": 77476,
    "Pfizer": 78003, "Ulta Beauty": 1403568, "Verizon": 732712, "Walmart": 104169,
}


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    time.sleep(0.25)
    return data


def all_filings(cik: int) -> tuple[str, list[dict]]:
    sub = json.loads(get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json"))
    blocks = [sub["filings"]["recent"]] + [json.loads(get(f"https://data.sec.gov/submissions/{f['name']}"))
                                           for f in sub["filings"].get("files", [])]
    rows = []
    for b in blocks:
        for form, acc, doc, rep, filed in zip(b["form"], b["accessionNumber"], b["primaryDocument"],
                                              b["reportDate"], b["filingDate"]):
            rows.append({"form": form, "acc": acc, "doc": doc, "report": rep, "filed": filed})
    return sub["name"], rows


def fiscal_year(period_end: str) -> int:
    """52/53-week years that end in the first week of January belong to the previous fiscal year
    (e.g. Johnson & Johnson's FY2022 ends 2023-01-01)."""
    y, m, d = (int(x) for x in period_end.split("-"))
    return y - 1 if m == 1 and d <= 7 else y


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in FB.read_text(encoding="utf-8").splitlines()]
    docs = sorted({(r["company"], r["doc_name"], int(r["doc_period"])) for r in rows if r["doc_type"] == "10k"})
    cache, found = {}, []
    for company, doc_name, year in docs:
        path = OUT / f"{doc_name}.htm"
        if company not in cache:
            cache[company] = all_filings(CIK[company])
        edgar_name, filings = cache[company]
        hits = [f for f in filings if f["form"] == "10-K" and fiscal_year(f["report"]) == year]
        if not hits:
            print(f"  !! no EDGAR 10-K for {doc_name} ({edgar_name})")
            continue
        f = hits[0]
        url = f"https://www.sec.gov/Archives/edgar/data/{CIK[company]}/{f['acc'].replace('-', '')}/{f['doc']}"
        if not path.exists():
            path.write_bytes(get(url))
        found.append({"company": company, "edgar_name": edgar_name, "cik": CIK[company], "doc_name": doc_name,
                      "fiscal_year": year, "period_end": f["report"], "filed": f["filed"], "url": url})
        print(f"  {doc_name:32s} {edgar_name[:28]:28s} period {f['report']}  filed {f['filed']}")
    (ROOT / "data" / "financebench" / "filings.json").write_text(json.dumps(found, indent=1))
    print(f"{len(found)}/{len(docs)} FinanceBench 10-K filings located on EDGAR")


if __name__ == "__main__":
    main()
