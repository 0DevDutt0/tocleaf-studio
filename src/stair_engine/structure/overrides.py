"""Explicit, recorded corrections to an outline before validation.

A correction file lists outline entries to exclude. Each rule names the entry by its displayed title and its
1-based destination page, as printed by `tocleaf analyze`, and gives a reason. Excluding an entry also excludes
everything nested under it. Every applied correction is written to the validation report and the dataset
manifest. A rule that matches no outline entry is an error, so a typo cannot pass silently.

Nothing here repairs numbering or renames titles. The tool only removes entries the user has named.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from stair_engine.structure.models import ValidationIssue
from stair_engine.structure.tree import RawHeading, split_number


@dataclass(frozen=True)
class ExcludeRule:
    title: str
    page: int  # 1-based destination page, as shown by `tocleaf analyze`
    reason: str


@dataclass(frozen=True)
class LineRule:
    pattern: str  # regular expression that must match the whole stripped line
    reason: str
    regex: re.Pattern = None  # type: ignore[assignment]


@dataclass
class CorrectionSet:
    description: str = ""
    exclude: list[ExcludeRule] = field(default_factory=list)
    remove_lines: list[LineRule] = field(default_factory=list)


def load_corrections(path: str | Path) -> CorrectionSet:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: the correction file must be a JSON object")
    unknown = set(data) - {"description", "exclude", "remove_lines"}
    if unknown:
        raise ValueError(
            f"{path}: unknown keys {sorted(unknown)}; allowed keys are 'description', 'exclude' and 'remove_lines'"
        )
    rules: list[ExcludeRule] = []
    for index, item in enumerate(data.get("exclude", [])):
        if not isinstance(item, dict):
            raise ValueError(f"{path}: exclude[{index}] must be an object")
        title = item.get("title")
        page = item.get("page")
        reason = item.get("reason")
        if not isinstance(title, str) or not title.strip():
            raise ValueError(f"{path}: exclude[{index}] needs a non-empty 'title'")
        if not isinstance(page, int) or page < 1:
            raise ValueError(f"{path}: exclude[{index}] needs 'page' as a 1-based integer")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"{path}: exclude[{index}] needs a 'reason'; every correction must say why")
        rules.append(ExcludeRule(title=title.strip(), page=page, reason=reason.strip()))
    line_rules: list[LineRule] = []
    for index, item in enumerate(data.get("remove_lines", [])):
        if not isinstance(item, dict):
            raise ValueError(f"{path}: remove_lines[{index}] must be an object")
        pattern = item.get("pattern")
        reason = item.get("reason")
        if not isinstance(pattern, str) or not pattern:
            raise ValueError(f"{path}: remove_lines[{index}] needs a non-empty 'pattern'")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"{path}: remove_lines[{index}] needs a 'reason'; every correction must say why")
        line_rules.append(LineRule(pattern=pattern, reason=reason.strip(), regex=re.compile(pattern)))
    return CorrectionSet(
        description=str(data.get("description", "")), exclude=rules, remove_lines=line_rules
    )


def _matches(rule: ExcludeRule, heading: RawHeading) -> bool:
    if heading.page_index + 1 != rule.page:
        return False
    _, displayed = split_number(heading.title)
    wanted = " ".join(rule.title.split()).lower()
    candidates = {" ".join(displayed.split()).lower(), " ".join(heading.title.split()).lower()}
    return wanted in candidates


def apply_corrections(
    headings: list[RawHeading], rules: list[ExcludeRule]
) -> tuple[list[int], list[dict], list[ValidationIssue]]:
    """Remove the named entries and everything nested under them.

    Returns (indices of kept headings, applied log, issues). Outline entries are in document order, so the
    entries nested under an entry are the contiguous run that follows it with a deeper level. Excluded entries
    are still located in the text by the caller, so that their text never merges into a neighbouring section.
    """

    matched_rules: set[int] = set()
    for index, rule in enumerate(rules):
        if any(_matches(rule, heading) for heading in headings):
            matched_rules.add(index)

    kept: list[int] = []
    applied: list[dict] = []
    position = 0
    while position < len(headings):
        heading = headings[position]
        rule_index = next((k for k, rule in enumerate(rules) if _matches(rule, heading)), None)
        if rule_index is None:
            kept.append(position)
            position += 1
            continue
        end = position + 1
        while end < len(headings) and headings[end].level > heading.level:
            end += 1
        nested_rules = sorted(
            {
                k
                for nested in headings[position + 1 : end]
                for k, rule in enumerate(rules)
                if _matches(rule, nested)
            }
        )
        rule = rules[rule_index]
        applied.append(
            {
                "action": "exclude",
                "title": heading.title,
                "page": rule.page,
                "reason": rule.reason,
                "level": heading.level,
                "index": position,
                "last_index": end - 1,
                "removed_entries": end - position,
                "nested_rules_inside_removed_entry": [rules[k].title for k in nested_rules],
            }
        )
        position = end

    issues = [
        ValidationIssue(
            code="CORRECTION_NOT_MATCHED",
            severity="error",
            message=(
                f"Correction '{rule.title}' on page {rule.page} matches no outline entry. "
                "Check the title and page as printed by `tocleaf analyze`."
            ),
        )
        for index, rule in enumerate(rules)
        if index not in matched_rules
    ]
    return kept, applied, issues
