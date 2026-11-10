"""Structure rules: printed numbers, generated numbers, hierarchy checks and running-text removal."""

from __future__ import annotations

from pathlib import Path

import pytest

from stair_engine.structure.build import analyze_pdf
from stair_engine.structure.pdf_reader import remove_running_text
from stair_engine.structure.tree import RawHeading, build_tree, check_leaf_labels, split_number

FIXTURE = Path(__file__).parent / "fixtures" / "irrigation_handbook.pdf"


def _codes(issues):
    return {issue.code for issue in issues}


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("1.2.1 Field Capacity", ("1.2.1", "Field Capacity")),
        ("Chapter 3: Scheduling and Economics", ("3", "Scheduling and Economics")),
        ("2. What is Action Research?", ("2", "What is Action Research?")),
        ("Water Sources", (None, "Water Sources")),
        ("2020 Report", ("2020", "Report")),
    ],
)
def test_split_number(title, expected):
    assert split_number(title) == expected


def test_generated_numbers_follow_position_when_none_printed():
    headings = [
        RawHeading("Alpha", 0, 0),
        RawHeading("Beta", 1, 0),
        RawHeading("Gamma", 1, 0),
        RawHeading("Delta", 0, 1),
    ]
    nodes, issues = build_tree(headings, page_count=2)
    assert [node.section_num for node in nodes] == ["1", "1.1", "1.2", "2"]
    assert [node.is_leaf for node in nodes] == [False, True, True, True]
    assert issues == []


def test_printed_numbers_are_kept_and_checked():
    headings = [
        RawHeading("1 Intro", 0, 0),
        RawHeading("1.1 Scope", 1, 0),
        RawHeading("2 Method", 0, 1),
    ]
    nodes, issues = build_tree(headings, page_count=2)
    assert [node.section_num for node in nodes] == ["1", "1.1", "2"]
    assert [node.title for node in nodes] == ["Intro", "Scope", "Method"]
    assert issues == []


def test_printed_number_outside_parent_is_an_error():
    headings = [RawHeading("1 Intro", 0, 0), RawHeading("2.1 Scope", 1, 0)]
    _, issues = build_tree(headings, page_count=1)
    assert "NUMBERING_INCONSISTENT" in _codes(issues)


def test_printed_numbers_out_of_order_are_an_error():
    headings = [RawHeading("2 Later", 0, 0), RawHeading("1 Earlier", 0, 0)]
    _, issues = build_tree(headings, page_count=1)
    assert "NUMBERING_OUT_OF_ORDER" in _codes(issues)


def test_skipped_outline_level_is_an_error_not_repaired():
    headings = [RawHeading("Chapter", 0, 0), RawHeading("Deep section", 2, 0)]
    nodes, issues = build_tree(headings, page_count=1)
    assert "INVALID_NESTING" in _codes(issues)
    assert len(nodes) == 2  # reported, not dropped


def test_page_order_and_bounds_are_checked():
    headings = [RawHeading("A", 0, 2), RawHeading("B", 0, 1), RawHeading("C", 0, 9)]
    _, issues = build_tree(headings, page_count=3)
    codes = _codes(issues)
    assert "OUT_OF_ORDER_PAGES" in codes
    assert "INVALID_PAGE_RANGE" in codes


def test_duplicate_labels_are_detected_for_corrected_trees():
    headings = [RawHeading("Same", 0, 0), RawHeading("Same", 0, 0)]
    nodes, _ = build_tree(headings, page_count=1)
    nodes[1].section_num = nodes[0].section_num
    assert "DUPLICATE_LABEL" in _codes(check_leaf_labels(nodes))


def test_running_text_is_removed_and_reported():
    pages = [f"Book Name\nBody text {i} words here\n{i + 1}" for i in range(4)]
    cleaned, report = remove_running_text(pages)
    assert all("Book Name" not in page for page in cleaned)
    assert report.removed_lines["book name"] == 4
    assert report.page_number_lines_removed == 4


def test_protected_heading_is_never_removed_as_running_text():
    pages = ["Summary\nText one here", "Summary\nText two here", "Summary\nText three here"]
    cleaned, report = remove_running_text(pages, protected_titles=["Summary"])
    assert all("Summary" in page for page in cleaned)
    assert report.removed_lines == {}


def test_fixture_structure_matches_design():
    structure = analyze_pdf(FIXTURE)
    assert structure.passed, [issue.to_dict() for issue in structure.errors]
    assert len(structure.nodes) == 19
    assert len(structure.leaves) == 13
    labels = {f"{node.section_num} {node.title}" for node in structure.leaves}
    assert "1.2.1 Field Capacity" in labels
    assert "2.1 Evapotranspiration" in labels
    evapo = next(node for node in structure.nodes if node.title == "Evapotranspiration")
    # Zero-based pages: the section starts on page 1 and runs onto page 2.
    assert evapo.page_start == 0 and evapo.page_end == 1
    assert all(node.heading_located for node in structure.nodes)
    assert "RUNNING_TEXT_REMOVED" in _codes(structure.issues)
    parent = next(node for node in structure.nodes if node.title == "Soil Moisture")
    assert not parent.is_leaf and parent.text  # parents keep their own intro text
    assert structure.breadcrumb(next(n.id for n in structure.nodes if n.title == "Field Capacity")) == (
        "Soil Water Foundations > Soil Moisture > Field Capacity"
    )


def test_pdf_without_outline_blocks_structure(tmp_path):
    from reportlab.pdfgen import canvas

    path = tmp_path / "flat.pdf"
    pdf = canvas.Canvas(str(path))
    pdf.drawString(72, 700, "Plain text with no bookmarks at all, long enough to be body text.")
    pdf.save()
    structure = analyze_pdf(path)
    assert not structure.passed
    assert "NO_NATIVE_TOC" in _codes(structure.issues)
