"""The exact McNemar test, and the paired comparison of two per-query files in both formats."""

from __future__ import annotations

import json

import pytest

from stair_engine.evaluation.significance import compare, exact_mcnemar_two_sided, top1_correct


@pytest.mark.parametrize(
    ("only_a", "only_b", "expected"),
    [(0, 10, 2 * (1 / 1024)), (8, 2, 2 * (56 / 1024)), (5, 5, 1.0), (0, 0, 1.0)],
)
def test_exact_p_values_match_the_binomial_tail(only_a, only_b, expected):
    assert exact_mcnemar_two_sided(only_a, only_b) == pytest.approx(expected, abs=1e-9)


def _write(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_compare_counts_discordant_queries_across_stair_and_bm25_formats(tmp_path):
    gold = ["n1"]
    stair = tmp_path / "stair" / "per_query.jsonl"
    bm25 = tmp_path / "bm25" / "bm25_per_query.jsonl"
    stair.parent.mkdir()
    bm25.parent.mkdir()
    _write(
        stair,
        [
            {"query_id": "q1", "gold": gold, "modes": {"constrained": {"ranked": ["n1", "n2"]}}},  # A right
            {"query_id": "q2", "gold": gold, "modes": {"constrained": {"ranked": ["n2", "n1"]}}},  # A wrong
            {"query_id": "q3", "gold": gold, "modes": {"constrained": {"ranked": ["n1"]}}},  # A right
            {"query_id": "q4", "gold": gold, "modes": {"constrained": {"ranked": ["n3"]}}},  # A wrong
            {"query_id": "q5", "gold": gold, "modes": {"constrained": {"ranked": ["n1"]}}},  # A right
        ],
    )
    _write(
        bm25,
        [
            {"query_id": "q1", "gold": gold, "ranked": ["n2", "n1"]},  # B wrong
            {"query_id": "q2", "gold": gold, "ranked": ["n1"]},  # B right
            {"query_id": "q3", "gold": gold, "ranked": ["n1"]},  # B right
            {"query_id": "q4", "gold": gold, "ranked": ["n3"]},  # B wrong
            {"query_id": "q5", "gold": gold, "ranked": ["n4"]},  # B wrong
        ],
    )

    report = compare(stair, bm25, mode="constrained")

    assert report["queries"] == 5
    assert report["top1_A"] == 0.6
    assert report["top1_B"] == 0.4
    assert report["A_right_B_wrong"] == 2  # q1 and q5
    assert report["A_wrong_B_right"] == 1  # q2
    assert report["net_queries_A_minus_B"] == 1
    assert report["mcnemar_exact_p_two_sided"] == 1.0  # 2 against 1 is not evidence


def test_a_multi_mode_file_needs_a_mode(tmp_path):
    path = tmp_path / "per_query.jsonl"
    _write(path, [{"query_id": "q1", "gold": ["n1"], "modes": {"constrained": {"ranked": ["n1"]}}}])
    with pytest.raises(ValueError, match="several decoding modes"):
        top1_correct(path)


def test_compare_refuses_different_query_sets(tmp_path):
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    _write(a, [{"query_id": "q1", "gold": ["n1"], "ranked": ["n1"]}])
    _write(b, [{"query_id": "q2", "gold": ["n1"], "ranked": ["n1"]}])
    with pytest.raises(ValueError, match="query sets differ"):
        compare(a, b)
