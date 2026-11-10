"""Write a training dataset in the layout the upstream trainer reads, plus a manifest.

Files written to ``out_dir``:

  toc.json          upstream-compatible ToC (canonical ids, all nodes, leaf flags)
  structure.json    complete canonical structure, text and validation report
  docs.jsonl        one record per ToC node with its own text span
  qa_candidates.jsonl   every accepted query with provenance, before splitting
  train.jsonl / val.jsonl / test.jsonl   records with ``question`` and ``reference``
  manifest.json     fingerprint, split configuration, leaf coverage, generator stats

Split protocol: stratified by leaf. Every leaf with at least two queries keeps at least one
query in train, because STAIR learns each leaf from training queries. A split by leaf would
hide every test leaf from training and measure something else. The fixed seed makes the split
reproducible, and the manifest records the fingerprint so runs can be compared.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from stair_engine.structure.build import to_upstream_toc
from stair_engine.structure.models import StructureResult


@dataclass(frozen=True)
class SplitConfig:
    train: float = 0.8
    val: float = 0.1
    test: float = 0.1
    seed: int = 42

    def __post_init__(self) -> None:
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"split fractions must sum to 1.0, got {total}")


def allocate_counts(n: int, config: SplitConfig) -> tuple[int, int, int]:
    """Return (train, val, test) counts for a leaf that has n queries."""

    if n == 1:
        return 1, 0, 0
    if n == 2:
        return 1, 0, 1
    test = max(1, round(n * config.test))
    val = max(1, round(n * config.val)) if n >= 5 else 0
    train = n - test - val
    if train < 1:
        train, test, val = 1, n - 1 - val, val
    return train, val, test


def stratified_split(
    records: list[dict], config: SplitConfig
) -> dict[str, list[dict]]:
    by_leaf: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_leaf[record["reference"][0]].append(record)

    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for leaf_id in sorted(by_leaf):
        items = sorted(by_leaf[leaf_id], key=lambda record: record["query_id"])
        # Seeded per leaf, so adding a leaf never reshuffles the others.
        random.Random(f"{config.seed}:{leaf_id}").shuffle(items)
        n_train, n_val, n_test = allocate_counts(len(items), config)
        splits["train"].extend(items[:n_train])
        splits["val"].extend(items[n_train : n_train + n_val])
        splits["test"].extend(items[n_train + n_val : n_train + n_val + n_test])
    return splits


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_canonical_json(row) + "\n")


def build_dataset(
    structure: StructureResult,
    accepted_queries: list[dict],
    generation_stats: dict,
    out_dir: str | Path,
    *,
    split: SplitConfig = SplitConfig(),
    generator_info: dict | None = None,
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    leaf_ids = {node.id for node in structure.leaves}
    for query in accepted_queries:
        if query["reference"][0] not in leaf_ids:
            raise ValueError(f"query references a non-leaf id: {query['reference']}")

    records = []
    for query_id, query in enumerate(sorted(accepted_queries, key=lambda q: (q["reference"][0], q["question"]))):
        records.append(
            {
                "query_id": f"q{query_id:05d}",
                "question": query["question"],
                "reference": list(query["reference"]),
                "answer": query["answer"],
                "question_type": query["question_type"],
            }
        )
    provenance_by_query = {
        f"q{index:05d}": query["provenance"]
        for index, query in enumerate(
            sorted(accepted_queries, key=lambda q: (q["reference"][0], q["question"]))
        )
    }

    splits = stratified_split(records, split)

    toc = to_upstream_toc(structure)
    (out / "toc.json").write_text(json.dumps(toc, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "structure.json").write_text(
        json.dumps(structure_payload(structure), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_jsonl(out / "docs.jsonl", docs_rows(structure))
    _write_jsonl(
        out / "qa_candidates.jsonl",
        [{**record, "provenance": provenance_by_query[record["query_id"]]} for record in records],
    )
    for name in ("train", "val", "test"):
        _write_jsonl(out / f"{name}.jsonl", splits[name])

    coverage: dict[str, dict[str, int]] = {}
    for leaf in structure.leaves:
        coverage[leaf.id] = {"section": f"{leaf.section_num} {leaf.title}", "train": 0, "val": 0, "test": 0}
    for name in ("train", "val", "test"):
        for record in splits[name]:
            coverage[record["reference"][0]][name] += 1
    # A leaf with no training query cannot be learned, so STAIR could never return it.
    uncovered = [
        {"leaf_id": leaf_id, "section": row["section"]}
        for leaf_id, row in coverage.items()
        if row["train"] == 0
    ]

    fingerprint_payload = {
        "toc": toc,
        "splits": {name: splits[name] for name in ("train", "val", "test")},
        "split_config": {"train": split.train, "val": split.val, "test": split.test, "seed": split.seed},
    }
    fingerprint = hashlib.sha256(_canonical_json(fingerprint_payload).encode("utf-8")).hexdigest()

    manifest = {
        "dataset_id": f"{Path(structure.source_path).stem}-{fingerprint[:12]}",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "fingerprint_sha256": fingerprint,
        "source": {
            "path": structure.source_path,
            "sha256": structure.source_sha256,
            "pages": structure.page_count,
            "book_title": structure.book_title,
        },
        "structure": {
            "nodes": len(structure.nodes),
            "leaves": len(structure.leaves),
            "errors": len(structure.errors),
            "validation_passed": structure.passed,
            "issue_counts": _issue_counts(structure),
        },
        "generation": {**(generator_info or {}), "stats": generation_stats},
        "split": {
            "strategy": "stratified_per_leaf",
            "train": split.train,
            "val": split.val,
            "test": split.test,
            "seed": split.seed,
        },
        "counts": {name: len(splits[name]) for name in ("train", "val", "test")} | {"total": len(records)},
        "leaf_coverage": coverage,
        "coverage_passed": not uncovered,
        "uncovered_leaves": uncovered,
        "targets": {
            "unit": "leaf",
            "identifier": "canonical node id (toc.json 'id')",
            "completion_format": "upstream prepare_input_output_stair: '\\n[\"<section_num> <title>\"]'",
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def structure_payload(structure: StructureResult) -> dict:
    return {
        "source": {
            "path": structure.source_path,
            "sha256": structure.source_sha256,
            "pages": structure.page_count,
            "book_title": structure.book_title,
        },
        "nodes": [
            {**node.to_dict(), "text": node.text, "breadcrumb": structure.breadcrumb(node.id)}
            for node in structure.nodes
        ],
        "issues": [issue.to_dict() for issue in structure.issues],
        "unmapped_front_matter_words": structure.unmapped_front_matter_words,
    }


def docs_rows(structure: StructureResult) -> list[dict]:
    return [
        {
            "id": node.id,
            "section_num": node.section_num,
            "title": node.title,
            "breadcrumb": structure.breadcrumb(node.id),
            "leaf": node.is_leaf,
            "depth": node.depth,
            "page_start": node.page_start,
            "page_end": node.page_end,
            "text": node.text,
        }
        for node in structure.nodes
    ]


def _issue_counts(structure: StructureResult) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for issue in structure.issues:
        counts[f"{issue.severity}:{issue.code}"] += 1
    return dict(counts)
