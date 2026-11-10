"""Evaluation query sets.

Two sources are supported:

* the synthetic test split written by the dataset builder (``test.jsonl``), and
* an external query file (``--queries``) whose gold answers are section numbers, which are mapped to
  canonical leaf ids through ``toc.json``.

Out-of-scope queries (``gold_section`` is null) are returned separately. They are not retrieval
targets, so they never enter the ranking metrics.
"""

from __future__ import annotations

import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_eval_records(data_dir: str | Path, queries_path: str | Path | None = None) -> tuple[list[dict], list[dict], str]:
    """Return (answerable records, out-of-scope records, source label)."""

    data = Path(data_dir)
    if queries_path is None:
        records = read_jsonl(data / "test.jsonl")
        for record in records:
            record["qtype"] = record.get("question_type", "synthetic")
        return records, [], "synthetic_test_split"

    toc = json.loads((data / "toc.json").read_text(encoding="utf-8"))
    leaf_by_section: dict[str, str] = {}
    for node in toc["table_of_contents"]:
        if node["leaf"]:
            leaf_by_section[node["section_num"]] = node["id"]
    known_sections = {node["section_num"] for node in toc["table_of_contents"]}

    answerable: list[dict] = []
    out_of_scope: list[dict] = []
    for row in read_jsonl(Path(queries_path)):
        gold = row.get("gold_section")
        if gold is None:
            out_of_scope.append({"query_id": row["query_id"], "question": row["question"], "qtype": row.get("type", "out_of_scope")})
            continue
        if gold not in known_sections:
            raise ValueError(f"{row['query_id']} names section {gold}, which is not in this ToC")
        if gold not in leaf_by_section:
            raise ValueError(f"{row['query_id']} names section {gold}, which is a parent node, not a leaf")
        answerable.append(
            {
                "query_id": row["query_id"],
                "question": row["question"],
                "reference": [leaf_by_section[gold]],
                "qtype": row.get("type", "unspecified"),
            }
        )
    return answerable, out_of_scope, f"author_queries:{Path(queries_path).name}"
