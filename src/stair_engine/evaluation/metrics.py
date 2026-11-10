"""Ranking metrics and output classification.

Failure types are kept apart on purpose, because they mean different things:

* invalid identifier: the generated text matches no ToC label at all
* non-leaf identifier: the text matches a ToC node that has children (the paper calls this a
  "hallucination"; this project does not, and reports it under its own name)
* wrong retrieval: the top result is a valid leaf that is not the gold leaf
* unparseable: the output is not a list of strings, so no identifier can be read from it

The paper's 0.05% figure measures only the non-leaf rate. It says nothing about final-answer
accuracy and must never be quoted as a hallucination rate for answers.
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass

from stair_engine.inference.contract import ToCView


def recall_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    return len(set(ranked[:k]) & gold) / len(gold)


def hit_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    return 1.0 if any(item in gold for item in ranked[:k]) else 0.0


def mrr_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    for position, item in enumerate(ranked[:k], start=1):
        if item in gold:
            return 1.0 / position
    return 0.0


def ndcg_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(position + 1) for position, item in enumerate(ranked[:k], start=1) if item in gold)
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(position + 1) for position in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def parse_identifier_list(text: str) -> list[str] | None:
    """Read a Python list of strings from generated text. Returns None when that is not possible."""

    start, end = text.find("["), text.rfind("]")
    if start == -1 or end <= start:
        return None
    try:
        parsed = ast.literal_eval(text[start : end + 1])
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return None
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        return None
    return parsed


@dataclass(frozen=True)
class ItemStatus:
    text: str
    status: str  # "valid_leaf" | "non_leaf" | "invalid_identifier"
    node_id: str | None


@dataclass(frozen=True)
class OutputClassification:
    parsed: bool
    items: tuple[ItemStatus, ...]
    ranked_leaf_ids: tuple[str, ...]  # distinct valid leaves, in output order

    @property
    def top_status(self) -> str:
        if not self.parsed or not self.items:
            return "unparseable"
        return self.items[0].status


def classify_output(text: str, view: ToCView) -> OutputClassification:
    parsed = parse_identifier_list(text)
    if parsed is None:
        return OutputClassification(parsed=False, items=(), ranked_leaf_ids=())
    items: list[ItemStatus] = []
    ranked: list[str] = []
    for label in parsed:
        node_id = view.id_by_label.get(label)
        if node_id is None:
            items.append(ItemStatus(text=label, status="invalid_identifier", node_id=None))
            continue
        if node_id in view.leaf_ids:
            items.append(ItemStatus(text=label, status="valid_leaf", node_id=node_id))
            if node_id not in ranked:
                ranked.append(node_id)
        else:
            items.append(ItemStatus(text=label, status="non_leaf", node_id=node_id))
    return OutputClassification(parsed=True, items=tuple(items), ranked_leaf_ids=tuple(ranked))


def summarise_queries(per_query: list[dict], ks: tuple[int, ...] = (1, 3, 5, 10)) -> dict:
    """Aggregate per-query records. Each record has 'ranked' (leaf ids), 'gold', 'top_status' and 'items'."""

    count = len(per_query)
    if count == 0:
        return {"queries": 0}

    summary: dict[str, float | int] = {"queries": count}
    for k in ks:
        summary[f"recall@{k}"] = sum(recall_at_k(r["ranked"], set(r["gold"]), k) for r in per_query) / count
        summary[f"hit@{k}"] = sum(hit_at_k(r["ranked"], set(r["gold"]), k) for r in per_query) / count
        summary[f"mrr@{k}"] = sum(mrr_at_k(r["ranked"], set(r["gold"]), k) for r in per_query) / count
        if k in (3, 5, 10):
            summary[f"ndcg@{k}"] = sum(ndcg_at_k(r["ranked"], set(r["gold"]), k) for r in per_query) / count

    total_items = sum(len(r["items"]) for r in per_query)
    status_counts: dict[str, int] = {"valid_leaf": 0, "non_leaf": 0, "invalid_identifier": 0}
    for record in per_query:
        for item in record["items"]:
            status_counts[item["status"]] = status_counts.get(item["status"], 0) + 1
    summary["unparseable_outputs"] = sum(1 for r in per_query if r["top_status"] == "unparseable")
    summary["items_total"] = total_items
    summary["invalid_identifier_rate"] = (status_counts["invalid_identifier"] / total_items) if total_items else 0.0
    summary["non_leaf_identifier_rate"] = (status_counts["non_leaf"] / total_items) if total_items else 0.0
    def top_node(record: dict) -> str | None:
        return record["items"][0]["node_id"] if record["items"] else None

    summary["top1_valid_leaf_rate"] = sum(1 for r in per_query if r["top_status"] == "valid_leaf") / count
    summary["wrong_retrieval_rate"] = sum(
        1
        for r in per_query
        if r["top_status"] == "valid_leaf" and top_node(r) not in set(r["gold"])
    ) / count
    summary["top1_correct_rate"] = sum(1 for r in per_query if top_node(r) in set(r["gold"])) / count
    return summary


def summarise_by_type(per_query: list[dict]) -> dict[str, dict]:
    """Summaries grouped by query type (for example direct, paraphrase, contextual, synthetic)."""

    groups: dict[str, list[dict]] = {}
    for record in per_query:
        groups.setdefault(record.get("qtype", "unspecified"), []).append(record)
    return {qtype: summarise_queries(rows) for qtype, rows in sorted(groups.items())}
