"""Builds data/false_premise/eval_set.json from scripts/false_premise_pairs.py and checks it
against the corpus: every evidence chunk must exist and belong to the stated company, and every
'absent' term must not occur in any chunk of that company (case-insensitive).

Usage:  python scripts/build_false_premise_set.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from false_premise_pairs import PAIRS  # noqa: E402

OUT = ROOT / "data" / "false_premise" / "eval_set.json"


def main() -> None:
    chunks = json.loads((ROOT / "data" / "chunk_texts.json").read_text(encoding="utf-8"))
    by_id = {c["chunk_id"]: c for c in chunks}
    by_company = defaultdict(list)
    for c in chunks:
        by_company[c["company"]].append(c["text"])
    items, problems = [], []
    for pid, company, ev, q_ok, q_fp, fp_type, absent in PAIRS:
        if ev not in by_id:
            problems.append(f"{pid}: evidence chunk {ev} missing")
            continue
        if by_id[ev]["company"] != company:
            problems.append(f"{pid}: evidence company {by_id[ev]['company']} != {company}")
        for term in absent:
            hits = sum(bool(re.search(re.escape(term), t, re.I)) for t in by_company[company])
            if hits:
                problems.append(f"{pid}: absent term {term!r} occurs in {hits} {company} chunks")
        base = {"pair_id": pid, "company": company, "evidence_chunk_id": ev,
                "evidence_text": by_id[ev]["text"]}
        items.append({**base, "id": f"{pid}_A", "question": q_ok, "label": "answerable", "fp_type": None})
        items.append({**base, "id": f"{pid}_F", "question": q_fp, "label": "false_premise",
                      "fp_type": fp_type, "absent_terms": absent})
    if problems:
        print("\n".join(problems))
        raise SystemExit(1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
    n_c = sum(i["fp_type"] == "contradicted" for i in items)
    print(f"wrote {len(items)} items ({len(PAIRS)} pairs: {n_c} contradicted, "
          f"{len(PAIRS) - n_c} absent) -> {OUT}")


if __name__ == "__main__":
    main()
