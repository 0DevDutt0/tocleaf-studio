"""Retrieval-side contracts: metrics, output classification, BM25, the leaf trie and upstream parity."""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import pytest

from stair_engine.evaluation.bm25 import BM25Index
from stair_engine.evaluation.metrics import (
    classify_output,
    hit_at_k,
    mrr_at_k,
    ndcg_at_k,
    parse_identifier_list,
    recall_at_k,
    summarise_queries,
)
from stair_engine.inference.constrained import ConstraintAudit, TokenTrie, make_prefix_allowed_tokens_fn, sequences_to_leaves
from stair_engine.inference.contract import ToCView, completion_text, load_toc_view


def _view() -> ToCView:
    label_by_id = {"p1": "1 Parent", "l1": "1.1 Alpha", "l2": "1.2 Beta"}
    id_by_label = {label: node_id for node_id, label in label_by_id.items()}
    return ToCView(
        book_title="Book",
        toc_text="\n".join(label_by_id.values()),
        label_by_id=label_by_id,
        id_by_label=id_by_label,
        leaf_ids=frozenset({"l1", "l2"}),
        is_leaf_by_id={"p1": False, "l1": True, "l2": True},
    )


def test_metrics_match_hand_computed_values():
    ranked = ["a", "b", "c", "d"]
    gold = {"b", "z"}
    assert recall_at_k(ranked, gold, 3) == pytest.approx(0.5)
    assert hit_at_k(ranked, gold, 1) == 0.0
    assert mrr_at_k(ranked, gold, 5) == pytest.approx(0.5)
    dcg = 1 / math.log2(3)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(ranked, gold, 3) == pytest.approx(dcg / idcg)


def test_parse_identifier_list_is_strict_about_strings():
    assert parse_identifier_list('["1.1 Alpha"]') == ["1.1 Alpha"]
    assert parse_identifier_list("[1, 2]") is None
    assert parse_identifier_list("no list here") is None


def test_classification_separates_invalid_non_leaf_and_leaf():
    view = _view()
    out = classify_output('["1 Parent", "9.9 Unknown", "1.2 Beta"]', view)
    assert [item.status for item in out.items] == ["non_leaf", "invalid_identifier", "valid_leaf"]
    assert out.ranked_leaf_ids == ("l2",)
    assert out.top_status == "non_leaf"
    assert classify_output("Sure! Here you go.", view).top_status == "unparseable"


def test_summary_counts_failure_types_separately():
    view = _view()
    per_query = []
    for text, gold in [('["1.1 Alpha"]', ["l1"]), ('["1 Parent"]', ["l1"]), ('["1.2 Beta"]', ["l1"])]:
        out = classify_output(text, view)
        per_query.append(
            {
                "ranked": list(out.ranked_leaf_ids),
                "gold": gold,
                "top_status": out.top_status,
                "items": [item.__dict__ for item in out.items],
            }
        )
    summary = summarise_queries(per_query)
    assert summary["non_leaf_identifier_rate"] == pytest.approx(1 / 3)
    assert summary["invalid_identifier_rate"] == 0.0
    assert summary["wrong_retrieval_rate"] == pytest.approx(1 / 3)
    assert summary["top1_correct_rate"] == pytest.approx(1 / 3)


def test_bm25_ranks_the_matching_leaf_first():
    index = BM25Index(
        {
            "water": "Irrigation water comes from reservoirs and wells.",
            "capacity": "Field capacity is the water a soil holds after drainage.",
            "cost": "Pumping cost depends on lift height.",
        }
    )
    assert index.rank("what is field capacity", k=3)[0] == "capacity"


def test_trie_allows_exactly_the_leaf_continuations():
    trie = TokenTrie({"a": [1, 2, 9], "b": [1, 3, 9]})
    assert trie.allowed_next([]) == [1]
    assert sorted(trie.allowed_next([1])) == [2, 3]
    assert trie.allowed_next([1, 2]) == [9]
    assert trie.leaf_for((1, 2, 9)) == "a"
    assert trie.leaf_for((1, 4, 9)) is None


def test_trie_rejects_prefix_completions():
    with pytest.raises(ValueError, match="prefix"):
        TokenTrie({"a": [1, 2], "b": [1, 2, 3]})


def test_constrained_mapping_reports_outside_trie_sequences():
    trie = TokenTrie({"a": [1, 2, 9]})
    audit = ConstraintAudit()
    leaves = sequences_to_leaves([[1, 2, 9, 0, 0], [5, 5, 9]], trie, eos_token_id=9, audit=audit)
    assert leaves == ["a", None]
    assert audit.sequences_outside_trie == 1


def test_prefix_function_only_offers_trie_continuations():
    import torch

    trie = TokenTrie({"a": [7, 8, 9]})
    audit = ConstraintAudit()
    fn = make_prefix_allowed_tokens_fn(trie, prefix_length=2, eos_token_id=9, audit=audit)
    ids = torch.tensor([100, 101, 7])
    assert fn(0, ids) == [8]


def test_completion_text_matches_upstream_format():
    assert completion_text(["1.1 Alpha", "1.2 Beta"]) == '\n["1.1 Alpha", "1.2 Beta"]'


def test_toc_loader_rejects_duplicate_labels(tmp_path):
    toc = {
        "title": "T",
        "table_of_contents": [
            {"id": "a", "section_num": "1", "title": "X", "leaf": True},
            {"id": "b", "section_num": "1", "title": "X", "leaf": True},
        ],
    }
    path = tmp_path / "toc.json"
    path.write_text(json.dumps(toc), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_toc_view(path)


UPSTREAM = Path(os.environ.get("STAIR_UPSTREAM", str(Path.home() / "stair-work" / "upstream" / "stair-clean")))


@pytest.mark.skipif(
    not (UPSTREAM / "src" / "stair" / "core" / "train" / "io_functions.py").exists(),
    reason="pinned upstream checkout not present (run scripts/setup_upstream.sh)",
)
def test_completion_and_prompts_match_the_pinned_upstream(tmp_path):
    """Run the upstream functions themselves (read from the pinned checkout, never copied here)."""

    from stair_engine.inference.contract import load_upstream_prompt_contract

    sys.path.insert(0, str(UPSTREAM / "src"))
    try:
        from stair.core import prompts as upstream_prompts
        from stair.core.train.io_functions import prepare_input_output_stair
    finally:
        sys.path.pop(0)

    contract = load_upstream_prompt_contract(UPSTREAM)
    assert contract.system_prompt == upstream_prompts.SYSTEM_PROMPT
    assert contract.user_template == upstream_prompts.USER_PROMPT

    toc_path = tmp_path / "toc.json"
    toc = {
        "title": "Contract Book",
        "table_of_contents": [
            {"id": "n0001", "section_num": "1", "title": "Intro", "leaf": False},
            {"id": "n0002", "section_num": "1.1", "title": "Alpha", "leaf": True},
            {"id": "n0003", "section_num": "1.2", "title": "Beta", "leaf": True},
        ],
    }
    toc_path.write_text(json.dumps(toc), encoding="utf-8")

    class FakeTokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            return "PROMPT:" + messages[1]["content"]

    for reference in (["n0002"], ["n0002", "n0003"]):
        example = {"question": "What is alpha?", "reference": reference}
        out = prepare_input_output_stair(
            data_config=None,
            examples=[example],
            tokenizer=FakeTokenizer(),
            split="train",
            system_prompt=upstream_prompts.SYSTEM_PROMPT,
            toc_json=str(toc_path),
            user_prompt=upstream_prompts.USER_PROMPT,
            output_col="{reference}",
        )
        labels = {"n0002": "1.1 Alpha", "n0003": "1.2 Beta"}
        assert out[0]["completion"] == completion_text([labels[r] for r in reference])
