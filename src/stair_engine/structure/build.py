"""Analyse a PDF into a validated, canonical ToC with per-section text and page ranges."""

from __future__ import annotations

from pathlib import Path

from stair_engine.structure.models import StructureResult, ValidationIssue
from stair_engine.structure.pdf_reader import char_to_page, locate_headings, read_pdf, remove_running_text
from stair_engine.structure.tree import build_tree, check_leaf_labels


def clean_section_text(raw: str) -> str:
    lines = [line.strip() for line in raw.replace("\r", "\n").split("\n")]
    return "\n".join(line for line in lines if line)


def analyze_pdf(
    path: str | Path,
    *,
    min_words_in_section: int = 5,
    book_title: str | None = None,
) -> StructureResult:
    content = read_pdf(path)
    stem_title = Path(path).stem.replace("_", " ").replace("-", " ").strip().title()
    title = book_title or content.metadata_title or stem_title

    issues: list[ValidationIssue] = [
        ValidationIssue(code="OUTLINE_ENTRY_WITHOUT_DESTINATION", severity="error", message=problem)
        for problem in content.outline_problems
    ]

    if not content.headings:
        issues.append(
            ValidationIssue(
                code="NO_NATIVE_TOC",
                severity="error",
                message=(
                    "The PDF has no usable outline (bookmarks). Heading inference is not "
                    "implemented yet, so this document cannot be structured automatically."
                ),
            )
        )
        return StructureResult(
            source_path=content.path,
            source_sha256=content.sha256,
            page_count=content.page_count,
            nodes=[],
            issues=issues,
            unmapped_front_matter_words=0,
            book_title=title,
        )

    nodes, tree_issues = build_tree(content.headings, content.page_count)
    issues.extend(tree_issues)

    cleaned_pages, running = remove_running_text(
        content.page_texts, protected_titles=[heading.title for heading in content.headings]
    )
    if running.removed_lines or running.page_number_lines_removed:
        removed_samples = "; ".join(
            f"'{line}' on {count} page(s)" for line, count in sorted(running.removed_lines.items(), key=lambda item: -item[1])[:5]
        )
        issues.append(
            ValidationIssue(
                code="RUNNING_TEXT_REMOVED",
                severity="warning",
                message=(
                    f"Removed repeated running text ({removed_samples or 'none'}) and "
                    f"{running.page_number_lines_removed} bare page-number line(s) before locating headings."
                ),
            )
        )

    joined, located, offsets = locate_headings(cleaned_pages, content.headings)
    located_indices = [index for index, span in enumerate(located) if span is not None]

    unmapped_front_matter_words = 0
    if located_indices:
        first_start = located[located_indices[0]].start  # type: ignore[union-attr]
        unmapped_front_matter_words = len(joined[:first_start].split())

    for position, index in enumerate(located_indices):
        node = nodes[index]
        span = located[index]
        assert span is not None
        if position + 1 < len(located_indices):
            next_start = located[located_indices[position + 1]].start  # type: ignore[union-attr]
        else:
            next_start = len(joined)
        node.text = clean_section_text(joined[span.end:next_start])
        if node.text:
            last_char = max(span.end, next_start - 1)
            node.page_end = max(node.page_start, char_to_page(last_char, offsets))

    for index, span in enumerate(located):
        if span is None:
            nodes[index].heading_located = False
            issues.append(
                ValidationIssue(
                    code="HEADING_NOT_FOUND",
                    severity="warning",
                    message=(
                        f"'{nodes[index].title}' (page {nodes[index].page_start + 1}) was not "
                        "found in the page text. Its section text is empty, so it will not "
                        "generate training questions."
                    ),
                    node_id=nodes[index].id,
                )
            )

    for node in nodes:
        if node.is_leaf and node.heading_located and node.word_count < min_words_in_section:
            issues.append(
                ValidationIssue(
                    code="EMPTY_SECTION",
                    severity="warning",
                    message=(
                        f"Leaf '{node.section_num} {node.title}' has {node.word_count} words "
                        f"(minimum {min_words_in_section}). It will not generate training questions."
                    ),
                    node_id=node.id,
                )
            )

    if unmapped_front_matter_words:
        issues.append(
            ValidationIssue(
                code="UNMAPPED_FRONT_MATTER",
                severity="warning",
                message=(
                    f"{unmapped_front_matter_words} words appear before the first outline "
                    "heading and belong to no section."
                ),
            )
        )

    issues.extend(check_leaf_labels(nodes))

    return StructureResult(
        source_path=content.path,
        source_sha256=content.sha256,
        page_count=content.page_count,
        nodes=nodes,
        issues=issues,
        unmapped_front_matter_words=unmapped_front_matter_words,
        book_title=title,
    )


def to_upstream_toc(result: StructureResult) -> dict:
    """The toc.json shape read by the upstream trainer and evaluator.

    The upstream loader reads ``id``, ``section_num`` and ``title``. Extra keys are
    ignored by it. ``id`` is the canonical node id, so the training target is never the
    generated text itself.
    """

    return {
        "title": result.book_title,
        "table_of_contents": [
            {
                "id": node.id,
                "section_num": node.section_num,
                "title": node.title,
                "leaf": node.is_leaf,
                "parent_id": node.parent_id,
                "depth": node.depth,
                "page_start": node.page_start,
                "page_end": node.page_end,
            }
            for node in result.nodes
        ],
    }
