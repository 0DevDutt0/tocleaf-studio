"""End-to-end steps: PDF -> validated structure -> synthetic queries -> dataset -> train -> evaluate.

Each step writes its output to disk and can be re-run on its own. A failed structure check stops
the pipeline before any model call. Nothing is generated for a ToC that has not passed validation.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from stair_engine.dataset.build import SplitConfig, build_dataset
from stair_engine.structure.build import analyze_pdf, to_upstream_toc
from stair_engine.structure.models import StructureResult
from stair_engine.synthetic.llm import ChatClient
from stair_engine.synthetic.qa import PROMPT_VERSION, generate_for_structure


class StructureValidationError(RuntimeError):
    """The ToC has validation errors. Training data is not generated for it."""


def write_validation_report(structure: StructureResult, path: Path) -> dict:
    report = {
        "source": structure.source_path,
        "source_sha256": structure.source_sha256,
        "pages": structure.page_count,
        "nodes": len(structure.nodes),
        "leaves": len(structure.leaves),
        "passed": structure.passed,
        "errors": [issue.to_dict() for issue in structure.errors],
        "warnings": [issue.to_dict() for issue in structure.issues if issue.severity == "warning"],
        "corrections": structure.corrections,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def build_dataset_from_pdf(
    pdf_path: str | Path,
    workspace: str | Path,
    *,
    client: ChatClient,
    questions_per_passage: int = 4,
    max_passage_words: int = 900,
    min_words_in_section: int = 5,
    split: SplitConfig = SplitConfig(),
    book_title: str | None = None,
    corrections=None,
) -> dict:
    work = Path(workspace)
    work.mkdir(parents=True, exist_ok=True)

    structure = analyze_pdf(
        pdf_path, min_words_in_section=min_words_in_section, book_title=book_title, corrections=corrections
    )
    report = write_validation_report(structure, work / "validation.json")
    if not structure.passed:
        raise StructureValidationError(
            f"{len(structure.errors)} structural error(s) block dataset generation. "
            f"See {work / 'validation.json'} and correct the ToC first."
        )

    accepted, stats = generate_for_structure(
        structure,
        client,
        questions_per_passage=questions_per_passage,
        max_passage_words=max_passage_words,
        min_words_in_section=min_words_in_section,
    )
    generator_info = {
        "model": client.model,
        "prompt_version": PROMPT_VERSION,
        "questions_per_passage": questions_per_passage,
        "max_passage_words": max_passage_words,
        "min_words_in_section": min_words_in_section,
        "warnings": [issue.message for issue in structure.issues if issue.severity == "warning"],
    }
    manifest = build_dataset(
        structure,
        accepted,
        stats.to_dict(),
        work / "dataset",
        split=split,
        generator_info=generator_info,
    )
    manifest["validation_warnings"] = report["warnings"]
    (work / "dataset" / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def evaluate_bm25(data_dir: str | Path, out_dir: str | Path, k: int = 10, queries_path: str | Path | None = None) -> dict:
    from stair_engine.evaluation.bm25 import BM25Index
    from stair_engine.evaluation.metrics import summarise_by_type, summarise_queries
    from stair_engine.evaluation.queries import load_eval_records
    from stair_engine.inference.contract import load_toc_view

    data = Path(data_dir)
    view = load_toc_view(data / "toc.json")
    structure = json.loads((data / "structure.json").read_text(encoding="utf-8"))
    documents: dict[str, str] = {}
    for node in structure["nodes"]:
        if node["leaf"]:
            documents[node["id"]] = f"{node['breadcrumb']}\n{node['text']}"
    index = BM25Index(documents)

    answerable, out_of_scope, query_source = load_eval_records(data, queries_path)
    per_query = []
    for record in answerable:
        ranked = index.rank(record["question"], k=k)
        per_query.append(
            {
                "query_id": record.get("query_id"),
                "qtype": record["qtype"],
                "question": record["question"],
                "gold": record["reference"],
                "ranked": ranked,
                "top_status": "valid_leaf" if ranked else "unparseable",
                "items": [{"text": view.label_by_id[leaf], "status": "valid_leaf", "node_id": leaf} for leaf in ranked],
            }
        )
    summary = {
        "method": "BM25 over canonical leaf sections (breadcrumb + section text)",
        "data_dir": str(data),
        "documents": len(documents),
        "query_source": query_source,
        "out_of_scope_queries": len(out_of_scope),
        "metrics": summarise_queries(per_query),
        "by_type": summarise_by_type(per_query),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "bm25_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(out / "bm25_per_query.jsonl", "w", encoding="utf-8") as handle:
        for row in per_query:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return summary


def _question_tokens(text: str) -> set[str]:
    import re

    from stair_engine.synthetic.qa import STOPWORDS

    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if token not in STOPWORDS}


def _jaccard(first: set[str], second: set[str]) -> float:
    if not first or not second:
        return 0.0
    return len(first & second) / len(first | second)


def augment_dataset(
    base_dataset_dir: str | Path,
    pdf_path: str | Path,
    out_dir: str | Path,
    *,
    client: ChatClient,
    corrections=None,
    questions_per_passage: int = 8,
    max_passage_words: int = 1500,
    leakage_threshold: float = 0.6,
    min_words_in_section: int = 5,
    book_title: str | None = None,
) -> dict:
    """Add synthetic training queries to an existing dataset without touching its validation or test splits.

    The document's structure must match the base dataset's ToC exactly, so that the new queries point at the
    same canonical leaf ids. A new query is rejected when it repeats an existing query, or when its content words
    overlap a validation or test query by at least ``leakage_threshold`` (Jaccard). The rejections are counted in
    the manifest.
    """

    from stair_engine.dataset.build import _write_jsonl
    from stair_engine.synthetic.qa import normalise_question

    base = Path(base_dataset_dir)
    out = Path(out_dir)
    base_manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    base_toc = json.loads((base / "toc.json").read_text(encoding="utf-8"))

    structure = analyze_pdf(
        pdf_path, min_words_in_section=min_words_in_section, book_title=book_title, corrections=corrections
    )
    if not structure.passed:
        raise StructureValidationError("the document's structure has validation errors; augmentation stopped")
    current_toc = to_upstream_toc(structure)
    if current_toc["table_of_contents"] != base_toc["table_of_contents"]:
        raise ValueError(
            "the document's structure differs from the base dataset's ToC (ids, labels or leaf flags). "
            "Augmentation would point new queries at the wrong leaves, so it was refused."
        )

    accepted, stats = generate_for_structure(
        structure,
        client,
        questions_per_passage=questions_per_passage,
        max_passage_words=max_passage_words,
        min_words_in_section=min_words_in_section,
    )

    def read(name: str) -> list[dict]:
        with open(base / f"{name}.jsonl", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    base_train, base_val, base_test = read("train"), read("val"), read("test")
    base_candidates = read("qa_candidates")
    eval_queries = base_val + base_test
    seen = {normalise_question(row["question"]) for row in base_train + base_val + base_test}
    eval_tokens = [_question_tokens(row["question"]) for row in eval_queries]

    rejected = {"duplicate_of_existing_query": 0, "near_duplicate_of_validation_or_test": 0}
    kept: list[dict] = []
    for query in accepted:
        key = normalise_question(query["question"])
        if key in seen:
            rejected["duplicate_of_existing_query"] += 1
            continue
        tokens = _question_tokens(query["question"])
        if any(_jaccard(tokens, other) >= leakage_threshold for other in eval_tokens):
            rejected["near_duplicate_of_validation_or_test"] += 1
            continue
        seen.add(key)
        kept.append(query)

    next_number = max(int(row["query_id"][1:]) for row in base_train + base_val + base_test + base_candidates) + 1
    new_train = []
    new_candidates = []
    for offset, query in enumerate(kept):
        query_id = f"q{next_number + offset:05d}"
        record = {
            "query_id": query_id,
            "question": query["question"],
            "reference": list(query["reference"]),
            "answer": query["answer"],
            "question_type": query["question_type"],
        }
        new_train.append(record)
        new_candidates.append({**record, "provenance": query["provenance"]})

    out.mkdir(parents=True, exist_ok=True)
    for name in ("toc.json", "structure.json", "docs.jsonl", "val.jsonl", "test.jsonl"):
        (out / name).write_bytes((base / name).read_bytes())
    _write_jsonl(out / "train.jsonl", base_train + new_train)
    _write_jsonl(out / "qa_candidates.jsonl", base_candidates + new_candidates)

    train_rows = base_train + new_train
    coverage: dict[str, dict] = {}
    for leaf_id, row in base_manifest["leaf_coverage"].items():
        coverage[leaf_id] = {"section": row["section"], "train": 0, "val": row["val"], "test": row["test"]}
    for row in train_rows:
        coverage[row["reference"][0]]["train"] += 1
    uncovered = [
        {"leaf_id": leaf_id, "section": row["section"]} for leaf_id, row in coverage.items() if row["train"] == 0
    ]

    import hashlib

    def canonical(value: object) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    fingerprint = hashlib.sha256(
        canonical({"toc": base_toc, "train": train_rows, "val": base_val, "test": base_test}).encode("utf-8")
    ).hexdigest()
    manifest = dict(base_manifest)
    manifest.update(
        {
            "dataset_id": f"{Path(structure.source_path).stem}-{fingerprint[:12]}",
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "fingerprint_sha256": fingerprint,
            "counts": {"train": len(train_rows), "val": len(base_val), "test": len(base_test), "total": len(train_rows) + len(base_val) + len(base_test)},
            "leaf_coverage": coverage,
            "coverage_passed": not uncovered,
            "uncovered_leaves": uncovered,
            "augmentation": {
                "base_dataset_id": base_manifest["dataset_id"],
                "base_fingerprint_sha256": base_manifest["fingerprint_sha256"],
                "added_train_queries": len(kept),
                "generated_candidates": len(accepted),
                "rejected": rejected,
                "leakage_threshold_jaccard": leakage_threshold,
                "questions_per_passage": questions_per_passage,
                "max_passage_words": max_passage_words,
                "model": client.model,
                "prompt_version": PROMPT_VERSION,
                "generation_stats": stats.to_dict(),
                "validation_and_test_unchanged": True,
            },
        }
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest
