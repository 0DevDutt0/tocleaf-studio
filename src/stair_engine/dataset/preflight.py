"""Check every training example against the sequence limit before any GPU time is spent.

The upstream trainer truncates long examples, and with TRL's default settings the truncation
removes the end of the completion, which is the part that carries the leaf identifier. This check
therefore refuses to continue when any example is longer than ``max_seq_length``, and it reports
the boundary-mismatch count that TRL warns about.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from stair_engine.inference.contract import (
    PromptContract,
    completion_text,
    load_toc_view,
    render_messages,
    training_token_layout,
)


class SequenceLengthError(ValueError):
    """Raised when at least one example would be truncated."""


def check_sequence_lengths(
    data_dir: str | Path,
    tokenizer,
    contract: PromptContract,
    max_seq_length: int,
) -> dict:
    data = Path(data_dir)
    view = load_toc_view(data / "toc.json")
    report: dict = {"max_seq_length": max_seq_length, "splits": {}}
    overflow: list[dict] = []
    all_lengths: list[int] = []
    mismatches = 0

    for split in ("train", "val", "test"):
        lengths: list[int] = []
        path = data / f"{split}.jsonl"
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                leaf_id = record["reference"][0]
                if leaf_id not in view.leaf_ids:
                    raise ValueError(f"{split} record {record.get('query_id')} targets a non-leaf {leaf_id}")
                prompt = tokenizer.apply_chat_template(
                    render_messages(contract, view, record["question"]), tokenize=False, add_generation_prompt=True
                )
                layout = training_token_layout(tokenizer, prompt, completion_text([view.label_by_id[leaf_id]]))
                total = len(layout.prefix_ids) + len(layout.target_ids)
                lengths.append(total)
                mismatches += int(layout.boundary_mismatch)
                if total > max_seq_length:
                    overflow.append({"split": split, "query_id": record.get("query_id"), "tokens": total})
        report["splits"][split] = {
            "examples": len(lengths),
            "min_tokens": min(lengths) if lengths else 0,
            "median_tokens": statistics.median(lengths) if lengths else 0,
            "max_tokens": max(lengths) if lengths else 0,
        }
        all_lengths.extend(lengths)

    report["toc_tokens"] = len(tokenizer(view.toc_text).input_ids)
    report["boundary_mismatch_examples"] = mismatches
    report["overflow_examples"] = len(overflow)
    report["max_tokens_overall"] = max(all_lengths) if all_lengths else 0
    report["passed"] = not overflow
    (data / "preflight.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if overflow:
        sample = ", ".join(f"{item['split']}:{item['query_id']}={item['tokens']}" for item in overflow[:5])
        raise SequenceLengthError(
            f"{len(overflow)} example(s) exceed max_seq_length={max_seq_length} "
            f"(e.g. {sample}). Raise max_seq_length in the training overrides, or shorten the ToC, "
            "before training. Nothing was truncated silently."
        )
    return report


class DatasetGateError(RuntimeError):
    """The dataset is not allowed to train: its structure or leaf coverage did not pass."""


def check_dataset_gates(data_dir: str | Path) -> dict:
    """Refuse training when the structure failed validation or a leaf has no training query."""

    manifest = json.loads((Path(data_dir) / "manifest.json").read_text(encoding="utf-8"))
    if not manifest["structure"]["validation_passed"]:
        raise DatasetGateError("the ToC has validation errors; correct the structure before training")
    if not manifest.get("coverage_passed", False):
        uncovered = manifest.get("uncovered_leaves", [])
        sample = ", ".join(item["section"] for item in uncovered[:5])
        raise DatasetGateError(
            f"{len(uncovered)} leaf section(s) have no training query (e.g. {sample}). Regenerate queries "
            "for them before training; STAIR cannot retrieve a section it never saw in training."
        )
    return manifest
