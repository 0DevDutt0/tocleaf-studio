"""Turn a flat, ordered list of outline headings into a validated ToC tree.

This module is pure Python so the hierarchy rules can be unit-tested without a PDF.
Nothing here repairs an invalid outline. Problems become ``ValidationIssue`` entries,
and training is blocked while any error remains.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from stair_engine.structure.models import (
    ORIGIN_PDF_OUTLINE,
    TocNode,
    ValidationIssue,
)


@dataclass(frozen=True)
class RawHeading:
    """One outline entry, in document order, as read from the PDF."""

    title: str
    level: int  # zero-based outline level as reported by the PDF
    page_index: int  # zero-based destination page


# "1.2.1 Field Capacity", "Chapter 3: Scheduling", "2. What is ..." -> (number, title)
_NUMBER_PREFIX = re.compile(
    r"^(?:(?:chapter|part|appendix|section)\s+)?(\d+(?:\.\d+)*)\s*[:.)\-–—]?\s+(.+)$",
    re.IGNORECASE,
)


def split_number(title: str) -> tuple[str | None, str]:
    """Separate a number the book printed in its heading from the heading text."""

    stripped = title.strip()
    match = _NUMBER_PREFIX.match(stripped)
    if match is None:
        return None, stripped
    return match.group(1), match.group(2).strip()


def _as_tuple(number: str) -> tuple[int, ...]:
    return tuple(int(part) for part in number.split("."))


def node_id_for(index: int) -> str:
    return f"n{index:04d}"


def build_tree(
    headings: list[RawHeading],
    page_count: int,
    origin: str = ORIGIN_PDF_OUTLINE,
) -> tuple[list[TocNode], list[ValidationIssue]]:
    """Build the ToC hierarchy and check it. Returns nodes in document order."""

    issues: list[ValidationIssue] = []
    nodes: list[TocNode] = []
    roots: list[TocNode] = []
    stack: list[tuple[int, TocNode]] = []  # (level, node) of the open ancestors
    previous_page = -1

    for index, heading in enumerate(headings):
        node_id = node_id_for(index)
        printed_number, title = split_number(heading.title)

        if not title:
            issues.append(
                ValidationIssue(
                    code="EMPTY_TITLE",
                    severity="error",
                    message=f"Outline entry {index} has an empty title.",
                    node_id=node_id,
                )
            )

        if heading.page_index < 0 or heading.page_index >= page_count:
            issues.append(
                ValidationIssue(
                    code="INVALID_PAGE_RANGE",
                    severity="error",
                    message=(
                        f"'{title}' points to page index {heading.page_index}, outside "
                        f"the document (0..{page_count - 1})."
                    ),
                    node_id=node_id,
                )
            )
        if heading.page_index < previous_page:
            issues.append(
                ValidationIssue(
                    code="OUT_OF_ORDER_PAGES",
                    severity="error",
                    message=(
                        f"'{title}' starts on page {heading.page_index + 1}, before the "
                        f"previous outline entry on page {previous_page + 1}."
                    ),
                    node_id=node_id,
                )
            )
        previous_page = max(previous_page, heading.page_index)

        while stack and stack[-1][0] >= heading.level:
            stack.pop()

        parent: TocNode | None = stack[-1][1] if stack else None
        expected_parent_level = heading.level - 1
        if heading.level > 0 and (not stack or stack[-1][0] != expected_parent_level):
            issues.append(
                ValidationIssue(
                    code="INVALID_NESTING",
                    severity="error",
                    message=(
                        f"'{title}' is at outline level {heading.level} but its nearest "
                        "ancestor is not at level "
                        f"{expected_parent_level}. The outline skips a level."
                    ),
                    node_id=node_id,
                )
            )

        siblings = parent.children if parent is not None else [node.id for node in roots]
        previous_sibling = None
        if siblings:
            previous_sibling = next(item for item in nodes if item.id == siblings[-1])

        if printed_number is not None:
            # The book's own number is authoritative. It must extend the parent's number and
            # must follow the previous sibling. Otherwise the outline is reported, not repaired.
            section_num = printed_number
            expected_depth = 1 if parent is None else len(parent.section_num.split(".")) + 1
            parent_prefix_ok = parent is None or printed_number.startswith(parent.section_num + ".")
            if len(printed_number.split(".")) != expected_depth or not parent_prefix_ok:
                issues.append(
                    ValidationIssue(
                        code="NUMBERING_INCONSISTENT",
                        severity="error",
                        message=(
                            f"'{title}' is printed as {printed_number} but sits under "
                            f"{parent.section_num if parent else 'the top level'}."
                        ),
                        node_id=node_id,
                    )
                )
            if previous_sibling is not None and _as_tuple(printed_number) <= _as_tuple(previous_sibling.section_num):
                issues.append(
                    ValidationIssue(
                        code="NUMBERING_OUT_OF_ORDER",
                        severity="error",
                        message=(
                            f"'{title}' is printed as {printed_number}, which does not come after "
                            f"{previous_sibling.section_num}."
                        ),
                        node_id=node_id,
                    )
                )
        elif parent is None:
            section_num = str(len(roots) + 1)
        else:
            section_num = f"{parent.section_num}.{len(parent.children) + 1}"

        node = TocNode(
            id=node_id,
            section_num=section_num,
            title=title,
            depth=len(section_num.split(".")),
            parent_id=parent.id if parent else None,
            page_start=heading.page_index,
            page_end=heading.page_index,
            origin=origin,
        )
        nodes.append(node)
        if parent is None:
            roots.append(node)
        else:
            parent.children.append(node.id)
        stack.append((heading.level, node))

    return nodes, issues


def check_leaf_labels(nodes: list[TocNode]) -> list[ValidationIssue]:
    """Labels must be unique. The model learns to emit 'section_num title'."""

    issues: list[ValidationIssue] = []
    seen_labels: dict[str, str] = {}
    for node in nodes:
        label = f"{node.section_num} {node.title}"
        if label in seen_labels:
            issues.append(
                ValidationIssue(
                    code="DUPLICATE_LABEL",
                    severity="error",
                    message=f"Label '{label}' is used by {seen_labels[label]} and {node.id}.",
                    node_id=node.id,
                )
            )
        else:
            seen_labels[label] = node.id
    return issues
