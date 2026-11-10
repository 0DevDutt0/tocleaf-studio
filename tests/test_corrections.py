"""Explicit corrections to an outline: rules, nesting, recorded effects, and no text leaking into neighbours."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stair_engine.structure.build import analyze_pdf
from stair_engine.structure.overrides import ExcludeRule, CorrectionSet, apply_corrections, load_corrections
from stair_engine.structure.tree import RawHeading

FIXTURE = Path(__file__).parent / "fixtures" / "irrigation_handbook.pdf"


def _headings():
    return [
        RawHeading("Front", 0, 0),
        RawHeading("Front detail", 1, 0),
        RawHeading("1 Chapter", 0, 1),
        RawHeading("1.1 Alpha", 1, 1),
        RawHeading("1.2 Beta", 1, 2),
        RawHeading("2 Next", 0, 3),
    ]


def test_excluding_an_entry_removes_its_nested_entries():
    rules = [ExcludeRule(title="Front", page=1, reason="front matter")]
    kept, applied, issues = apply_corrections(_headings(), rules)
    assert kept == [2, 3, 4, 5]
    assert applied[0]["removed_entries"] == 2
    assert applied[0]["last_index"] == 1
    assert issues == []


def test_a_rule_that_matches_nothing_is_an_error():
    rules = [ExcludeRule(title="Missing", page=9, reason="typo")]
    _, applied, issues = apply_corrections(_headings(), rules)
    assert applied == []
    assert [issue.code for issue in issues] == ["CORRECTION_NOT_MATCHED"]


def test_page_must_match_so_repeated_titles_are_not_ambiguous():
    headings = [RawHeading("Introduction", 0, 0), RawHeading("Introduction", 0, 5)]
    kept, applied, _ = apply_corrections(headings, [ExcludeRule(title="Introduction", page=6, reason="second only")])
    assert kept == [0]
    assert applied[0]["page"] == 6


def test_corrections_file_requires_a_reason(tmp_path):
    path = tmp_path / "corrections.json"
    path.write_text(json.dumps({"exclude": [{"title": "X", "page": 1}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="reason"):
        load_corrections(path)


def test_corrections_file_rejects_unknown_keys(tmp_path):
    path = tmp_path / "corrections.json"
    path.write_text(json.dumps({"rename": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown keys"):
        load_corrections(path)


def test_excluded_text_never_merges_into_the_previous_section():
    """Drainage (page 1) follows Permanent Wilting Point on the same page. Excluding Drainage must keep its
    text out of the wilting-point section, because the excluded entry still bounds that section."""

    structure = analyze_pdf(
        FIXTURE,
        corrections=CorrectionSet(exclude=[ExcludeRule(title="Drainage", page=1, reason="test exclusion")]),
    )
    assert structure.passed
    assert len(structure.leaves) == 12
    assert all("Drainage removes" not in node.text for node in structure.nodes)
    wilting = next(node for node in structure.nodes if node.title == "Permanent Wilting Point")
    assert "Drainage" not in wilting.text
    assert structure.corrections[0]["words_removed"] > 0


def test_an_unmatched_correction_blocks_validation():
    structure = analyze_pdf(
        FIXTURE,
        corrections=CorrectionSet(exclude=[ExcludeRule(title="No Such Section", page=1, reason="typo")]),
    )
    assert not structure.passed
    assert "CORRECTION_NOT_MATCHED" in {issue.code for issue in structure.errors}
