"""Build the v2 knowledge base and test set from real, open-source data.

Knowledge base (all SEC EDGAR 10-Ks, chunked with the unchanged src/data_pipeline.py code):
  * the FinanceBench 10-Ks of the knowledge-base (KB) companies, and
  * the 7 original 2025/26 big-tech 10-Ks (kept so the counterfactual stress test stays meaningful).

Test questions:
  IN_DISTRIBUTION  FinanceBench questions on a KB company's 10-K, whose evidence passage is verified
                   to be present in our chunked filing.
  NEAR_OOD         every FinanceBench question about a held-out company (none of its filings are indexed):
                   same domain, style and vocabulary, but unanswerable from the knowledge base.
  NEAR_OOD_ADV     the 100 counterfactual "entity perturbation" questions (adversarial stress test).
  FAR_OOD          the 100 Open Trivia Database questions.

Company split: companies are shuffled with a fixed seed and assigned to the held-out group until it
holds about one third of the FinanceBench 10-K questions.  Amazon and Microsoft always stay in the
knowledge base because their 2025/26 10-Ks are indexed.

Usage (from the project root):  python scripts/build_financebench_dataset.py
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data_pipeline import (CHUNK_OVERLAP, CHUNK_SIZE, RecursiveCharacterTextSplitter,  # noqa: E402
                               clean_lines, html_to_lines, is_informative)

DATA = ROOT / "data"
FBDIR = DATA / "financebench"
SOURCES = DATA / "sources"
SEED = 0
HELD_OUT_SHARE = 1 / 3
ALWAYS_KB = {"Amazon", "Microsoft"}
BIGTECH = [("AAPL", "Apple Inc."), ("MSFT", "Microsoft Corp."), ("GOOGL", "Alphabet Inc."), ("NVDA", "NVIDIA Corp."),
           ("AMZN", "Amazon.com Inc."), ("META", "Meta Platforms Inc."), ("TSLA", "Tesla Inc.")]
EVIDENCE_MIN_COVERAGE = 0.6


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def trigrams(ws: list[str]) -> set:
    return {tuple(ws[i:i + 3]) for i in range(len(ws) - 2)}


def evidence_coverage(evidence: list[dict], doc_tri: set) -> float:
    """Share of the evidence passage's word trigrams found in our extracted filing text (best passage)."""
    best = 0.0
    for e in evidence:
        tri = trigrams(words(e.get("evidence_text", "")))
        if tri:
            best = max(best, len(tri & doc_tri) / len(tri))
    return best


def parse_evidence(raw) -> list[dict]:
    if isinstance(raw, list):
        return raw
    import ast
    return ast.literal_eval(raw)


def main() -> None:
    fb = [json.loads(l) for l in (FBDIR / "financebench_merged.jsonl").read_text(encoding="utf-8").splitlines()]
    filings = {f["doc_name"]: f for f in json.loads((FBDIR / "filings.json").read_text())}

    # ---------------------------------------------------------------- company split
    q10k = Counter(r["company"] for r in fb if r["doc_type"] == "10k")
    companies = sorted(q10k)
    rng = np.random.default_rng(SEED)
    held_out, n_held = set(), 0
    for c in rng.permutation(companies):
        if c in ALWAYS_KB:
            continue
        if n_held + q10k[c] <= HELD_OUT_SHARE * sum(q10k.values()) + 2:
            held_out.add(c)
            n_held += q10k[c]
    kb_companies = sorted(set(companies) - held_out)
    print(f"held-out companies ({len(held_out)}): {sorted(held_out)}")

    # ---------------------------------------------------------------- knowledge base
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
                                              separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
                                              keep_separator="end")
    corpus, seen, doc_tri, doc_rows = [], set(), {}, []

    def add_doc(doc_id: str, company: str, source: str, html: str, year) -> None:
        text = "\n".join(clean_lines(html_to_lines(html)))
        doc_tri[doc_id] = trigrams(words(text))
        kept = 0
        for piece in splitter.split_text(text):
            piece = piece.strip()
            if len(piece) < 100 or not is_informative(piece) or piece in seen:
                continue
            seen.add(piece)
            corpus.append({"chunk_id": f"{doc_id}_chunk_{kept:05d}", "doc_name": doc_id, "company": company,
                           "year": str(year), "source": source, "text": piece, "char_count": len(piece)})
            kept += 1
        doc_rows.append({"doc_name": doc_id, "company": company, "year": year, "source": source, "chunks": kept})
        print(f"  {doc_id:32s} {kept:6,d} chunks", flush=True)

    for t, name in BIGTECH:
        add_doc(f"{t}_10K_latest", name, "Phase 1 big-tech 10-K (EDGAR)",
                (DATA / "raw" / f"{t}_10K.htm").read_text(encoding="utf-8", errors="ignore"), "2025/26")
    for doc_name, f in sorted(filings.items()):
        html = (DATA / "raw" / "fb" / f"{doc_name}.htm").read_text(encoding="utf-8", errors="ignore")
        if f["company"] in held_out:
            doc_tri[doc_name] = trigrams(words("\n".join(clean_lines(html_to_lines(html)))))  # for checks only
            continue
        add_doc(doc_name, f["company"], "FinanceBench 10-K (EDGAR)", html, f["fiscal_year"])

    # ---------------------------------------------------------------- questions
    queries, excluded, coverage = [], [], []
    for r in fb:
        base = {"query_text": r["question"], "source": "FinanceBench", "financebench_id": r["financebench_id"],
                "company": r["company"], "doc_name": r["doc_name"], "doc_type": r["doc_type"],
                "question_type": r["question_type"], "answer": r["answer"]}
        if r["company"] in held_out:
            queries.append({**base, "label": "NEAR_OOD", "expected_behavior": "REJECT_OOD"})
        elif r["doc_type"] != "10k":
            excluded.append({**base, "reason": "KB company, but the cited document is not a 10-K"})
        else:
            cov = evidence_coverage(parse_evidence(r["evidence"]), doc_tri[r["doc_name"]])
            coverage.append({"financebench_id": r["financebench_id"], "doc_name": r["doc_name"], "coverage": cov})
            if cov >= EVIDENCE_MIN_COVERAGE:
                queries.append({**base, "label": "IN_DISTRIBUTION", "expected_behavior": "PASS_TO_RETRIEVER",
                                "evidence_coverage": round(cov, 3)})
            else:
                excluded.append({**base, "reason": f"evidence not found in extracted filing (coverage {cov:.2f})"})
    # held-out filings must really be absent: check their evidence is not in the knowledge base by accident
    kb_tri = set().union(*(doc_tri[d["doc_name"]] for d in doc_rows))

    near_adv = json.loads((SOURCES / "near_ood_adversarial.json").read_text(encoding="utf-8"))
    far = json.loads((SOURCES / "far_ood_opentdb.json").read_text(encoding="utf-8"))
    for q in near_adv:
        queries.append({"query_text": q["query_text"], "label": "NEAR_OOD_ADV", "source": "Counterfactual (entity perturbation)",
                        "company": q["company_context"], "sub_category": q["sub_category"], "expected_behavior": "REJECT_OOD"})
    for q in far:
        queries.append({"query_text": q["query_text"], "label": "FAR_OOD", "source": "Open Trivia Database",
                        "category": q["category"], "expected_behavior": "REJECT_OOD"})
    prefix = {"IN_DISTRIBUTION": "ID", "NEAR_OOD": "NEAR", "NEAR_OOD_ADV": "ADV", "FAR_OOD": "FAR"}
    count = Counter()
    for q in queries:
        count[q["label"]] += 1
        q["query_id"] = f"{prefix[q['label']]}_{count[q['label']]:04d}"

    # ---------------------------------------------------------------- write
    (DATA / "chunk_texts.json").write_text(json.dumps(corpus, indent=1, ensure_ascii=False), encoding="utf-8")
    (DATA / "test_queries.json").write_text(json.dumps(queries, indent=1, ensure_ascii=False), encoding="utf-8")
    report = {
        "held_out_companies": sorted(held_out), "kb_companies": kb_companies + [n for _, n in BIGTECH],
        "documents": doc_rows, "n_chunks": len(corpus), "label_counts": dict(count),
        "excluded_questions": excluded, "evidence_coverage": coverage,
        "held_out_evidence_in_kb": [  # sanity: should all be low
            {"financebench_id": r["financebench_id"],
             "coverage": evidence_coverage(parse_evidence(r["evidence"]), kb_tri)}
            for r in fb if r["company"] in held_out],
    }
    (FBDIR / "dataset_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    leak = [x for x in report["held_out_evidence_in_kb"] if x["coverage"] >= EVIDENCE_MIN_COVERAGE]
    print(f"\nchunks: {len(corpus):,} from {len(doc_rows)} filings;  questions: {dict(count)}")
    print(f"excluded FinanceBench questions: {len(excluded)}  "
          f"({Counter(e['reason'].split(' (')[0] for e in excluded)})")
    print(f"held-out questions whose evidence nonetheless appears in the KB: {len(leak)}")


if __name__ == "__main__":
    main()
