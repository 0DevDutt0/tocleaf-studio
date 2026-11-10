"""Read a PDF's native outline (bookmarks) and page text.

Only the native outline is used here. A PDF without bookmarks yields an error, and no
heading inference is attempted. Heading inference is a later milestone and must be
reported as inferred, never as an original table of contents.
"""

from __future__ import annotations

import bisect
import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import pypdfium2 as pdfium

from stair_engine.structure.tree import RawHeading


# Separates page texts inside the joined document string. Heading searches never match
# across this boundary, because the whitespace-tolerant pattern cannot include it.
PAGE_SEPARATOR = "\n\n\f\n\n"


@dataclass
class PdfContent:
    path: str
    sha256: str
    page_count: int
    page_texts: list[str]
    headings: list[RawHeading]
    outline_problems: list[str] = field(default_factory=list)
    metadata_title: str = ""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _heading_pattern(title: str) -> re.Pattern[str]:
    words = title.split()
    if not words:
        raise ValueError("cannot build a heading pattern from an empty title")
    # Straight and typographic apostrophes differ between PDF producers, so either one matches.
    def escape_char(character: str) -> str:
        if character in "'’":
            return "['’]"
        return re.escape(character)

    escaped_words = ["".join(escape_char(character) for character in word) for word in words]
    return re.compile(r"\s+".join(escaped_words), re.IGNORECASE)


def read_pdf(path: str | Path) -> PdfContent:
    path = str(path)
    document = pdfium.PdfDocument(path)
    try:
        page_count = len(document)
        page_texts: list[str] = []
        for page_index in range(page_count):
            text_page = document[page_index].get_textpage()
            try:
                page_texts.append(text_page.get_text_range())
            finally:
                text_page.close()

        headings: list[RawHeading] = []
        outline_problems: list[str] = []
        for outline_index, item in enumerate(document.get_toc()):
            title = item.get_title().strip()
            destination = item.get_dest()
            page_index = destination.get_index() if destination is not None else None
            if page_index is None:
                outline_problems.append(
                    f"Outline entry {outline_index} ('{title}') has no page destination."
                )
                continue
            headings.append(RawHeading(title=title, level=int(item.level), page_index=int(page_index)))

        metadata = document.get_metadata_dict() or {}
        metadata_title = str(metadata.get("Title") or "").strip()
    finally:
        document.close()

    return PdfContent(
        path=path,
        sha256=sha256_file(path),
        page_count=page_count,
        page_texts=page_texts,
        headings=headings,
        outline_problems=outline_problems,
        metadata_title=metadata_title,
    )


_PAGE_NUMBER_LINE = re.compile(r"^(?:page\s+)?\d{1,4}(?:\s+of\s+\d{1,4})?$", re.IGNORECASE)


@dataclass
class RunningTextReport:
    removed_lines: dict[str, int]  # normalised line -> number of pages it was removed from
    page_number_lines_removed: int


def remove_running_text(
    page_texts: list[str], protected_titles: Iterable[str] = ()
) -> tuple[list[str], RunningTextReport]:
    """Remove running headers, running footers and bare page numbers.

    A line counts as running text when it is short and appears on at least a quarter of the pages
    (and at least three of them). Bare page numbers are removed on any page. Everything removed is
    reported, so the caller can show it. Nothing is removed silently.
    """

    page_line_sets: list[set[str]] = []
    for text in page_texts:
        lines = {line.strip() for line in text.split("\n") if line.strip()}
        page_line_sets.append({line.lower() for line in lines})

    protected = {" ".join(title.split()).lower() for title in protected_titles}
    # A running line appears on at least a quarter of the pages. Body boilerplate such as a repeated
    # 'Learning Objectives' label appears on far fewer pages, so it is kept.
    threshold = max(3, -(-len(page_texts) // 4))
    counts: dict[str, int] = {}
    for line_set in page_line_sets:
        for line in line_set:
            counts[line] = counts.get(line, 0) + 1
    running = {
        line
        for line, count in counts.items()
        if count >= threshold and len(line.split()) <= 12 and line not in protected
    }

    removed_running: dict[str, int] = {}
    page_numbers_removed = 0
    cleaned: list[str] = []
    for text in page_texts:
        kept: list[str] = []
        for raw_line in text.split("\n"):
            key = raw_line.strip().lower()
            if key and key in running:
                removed_running[key] = removed_running.get(key, 0) + 1
                continue
            if key and _PAGE_NUMBER_LINE.match(key):
                page_numbers_removed += 1
                continue
            kept.append(raw_line)
        cleaned.append("\n".join(kept))
    report = RunningTextReport(removed_lines=removed_running, page_number_lines_removed=page_numbers_removed)
    return cleaned, report


@dataclass(frozen=True)
class LocatedHeading:
    start: int
    end: int


def page_start_offsets(page_texts: list[str]) -> list[int]:
    offsets = []
    cursor = 0
    for index, text in enumerate(page_texts):
        offsets.append(cursor)
        cursor += len(text)
        if index < len(page_texts) - 1:
            cursor += len(PAGE_SEPARATOR)
    return offsets


def locate_headings(
    page_texts: list[str],
    headings: list[RawHeading],
    search_pages_after: int = 1,
) -> tuple[str, list[LocatedHeading | None], list[int]]:
    """Find each heading's text inside the joined document string.

    Headings are searched in outline order, starting at their destination page. The
    search may run ``search_pages_after`` pages further, to tolerate outlines that point at
    the page before a heading. A heading that cannot be found yields ``None``, which the
    caller reports. The search never moves backwards, so the located spans stay in order.
    """

    joined = PAGE_SEPARATOR.join(page_texts)
    offsets = page_start_offsets(page_texts)
    located: list[LocatedHeading | None] = []
    cursor = 0
    for heading in headings:
        if heading.page_index < 0 or heading.page_index >= len(page_texts):
            located.append(None)
            continue
        window_start = max(cursor, offsets[heading.page_index])
        last_page = min(heading.page_index + search_pages_after, len(page_texts) - 1)
        window_end = offsets[last_page] + len(page_texts[last_page])
        match = _heading_pattern(heading.title).search(joined, window_start, window_end)
        if match is None:
            located.append(None)
        else:
            located.append(LocatedHeading(start=match.start(), end=match.end()))
            cursor = match.end()
    return joined, located, offsets


def char_to_page(offset: int, offsets: list[int]) -> int:
    """Map a character offset in the joined string to a zero-based page index."""

    return max(0, bisect.bisect_right(offsets, offset) - 1)
