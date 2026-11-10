"""End-to-end steps: PDF -> validated structure -> synthetic queries -> dataset -> train -> evaluate.

Each step writes its output to disk and can be re-run on its own. A failed structure check stops
the pipeline before any model call. Nothing is generated for a ToC that has not passed validation.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from stair_engine.dataset.build import SplitConfig, build_dataset
from stair_engine.structure.build import analyze_pdf
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
