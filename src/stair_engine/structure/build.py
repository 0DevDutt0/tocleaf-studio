"""Analyse a PDF into a validated, canonical ToC with per-section text and page ranges."""

from __future__ import annotations

from pathlib import Path

from stair_engine.structure.models import StructureResult, ValidationIssue
from stair_engine.structure.overrides import CorrectionSet, apply_corrections
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
    corrections: CorrectionSet | None = None,
) -> StructureResult:
    content = read_pdf(path)
    stem_title = Path(path).stem.replace("_", " ").replace("-", " ").strip().title()
    title = book_title or content.metadata_title or stem_title

    issues: list[ValidationIssue] = [
        ValidationIssue(code="OUTLINE_ENTRY_WITHOUT_DESTINATION", severity="error", message=problem)
        for problem in content.outline_problems
    ]

    all_headings = content.headings
    kept_indices = list(range(len(all_headings)))
    applied_corrections: list[dict] = []
    if corrections is not None and corrections.exclude:
        kept_indices, applied_corrections, correction_issues = apply_corrections(all_headings, corrections.exclude)
        issues.extend(correction_issues)
    kept_headings = [all_headings[index] for index in kept_indices]

    if not kept_headings:
        message = (
            "The PDF has no usable outline (bookmarks). Heading inference is not implemented yet, so this "
            "document cannot be structured automatically."
            if not all_headings
            else "Every outline entry was excluded by the correction file, so no sections remain."
        )
        issues.append(ValidationIssue(code="NO_NATIVE_TOC" if not all_headings else "NO_SECTIONS_LEFT", severity="error", message=message))
        return StructureResult(
            source_path=content.path,
            source_sha256=content.sha256,
            page_count=content.page_count,
            nodes=[],
            issues=issues,
            unmapped_front_matter_words=0,
            book_title=title,
            corrections=applied_corrections,
        )

    nodes, tree_issues = build_tree(kept_headings, content.page_count)
    issues.extend(tree_issues)

    cleaned_pages, running = remove_running_text(
        content.page_texts, protected_titles=[heading.title for heading in all_headings]
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

    # Locate every outline entry, kept or excluded. Excluded entries still mark where the previous section
    # ends, so their text never merges into a neighbouring section.
    if corrections is not None:
        for rule in corrections.remove_lines:
            removed = 0
            kept_pages: list[str] = []
            for page_text in cleaned_pages:
                kept_lines = []
                for line in page_text.split("\n"):
                    if rule.regex.fullmatch(line.strip()):
                        removed += 1
                    else:
                        kept_lines.append(line)
                kept_pages.append("\n".join(kept_lines))
            cleaned_pages = kept_pages
            applied_corrections.append(
                {"action": "remove_lines", "pattern": rule.pattern, "reason": rule.reason, "lines_removed": removed}
            )
            if removed == 0:
                issues.append(
                    ValidationIssue(
                        code="CORRECTION_NOT_MATCHED",
                        severity="error",
                        message=f"Line rule {rule.pattern!r} matched no line in the document.",
                    )
                )

    joined, located, offsets = locate_headings(cleaned_pages, all_headings)
    located_all = [index for index, span in enumerate(located) if span is not None]

    def next_boundary(index: int) -> int:
        for other in located_all:
            if other > index:
                return located[other].start  # type: ignore[union-attr]
        return len(joined)

    unmapped_front_matter_words = 0
    if located_all:
        first_start = located[located_all[0]].start  # type: ignore[union-attr]
        unmapped_front_matter_words = len(joined[:first_start].split())

    for node_index, full_index in enumerate(kept_indices):
        node = nodes[node_index]
        span = located[full_index]
        if span is None:
            node.heading_located = False
            continue
        end_of_text = next_boundary(full_index)
        node.text = clean_section_text(joined[span.end : end_of_text])
        if node.text:
            last_char = max(span.end, end_of_text - 1)
            node.page_end = max(node.page_start, char_to_page(last_char, offsets))

    for node_index, full_index in enumerate(kept_indices):
        if located[full_index] is None:
            node = nodes[node_index]
            issues.append(
                ValidationIssue(
                    code="HEADING_NOT_FOUND",
                    severity="warning",
                    message=(
                        f"'{node.title}' (page {node.page_start + 1}) was not found in the page text. "
                        "Its section text is empty, so it will not generate training questions."
                    ),
                    node_id=node.id,
                )
            )

    for entry in applied_corrections:
        if entry["action"] != "exclude":
            continue
        first, last = entry.pop("index"), entry.pop("last_index")
        span = located[first]
        entry["words_removed"] = (
            len(joined[span.end : next_boundary(last)].split()) if span is not None else 0
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
        corrections=applied_corrections,
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
