"""Canonical document-structure model.

Every ToC node carries a stable canonical ``id`` that is independent of the text the
model generates. A node is a *leaf* when it has no child nodes in the ToC tree. That is
a property of the hierarchy, not of whether its text happens to be long or short.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


ORIGIN_PDF_OUTLINE = "pdf_outline"
ORIGIN_USER_CORRECTED = "user_corrected"


@dataclass
class TocNode:
    id: str
    section_num: str
    title: str
    depth: int
    parent_id: Optional[str]
    page_start: int
    page_end: int
    origin: str = ORIGIN_PDF_OUTLINE
    children: list[str] = field(default_factory=list)
    # Text owned by this node: for a parent, the text before its first child heading;
    # for a leaf, all text up to the next heading in document order.
    text: str = ""
    heading_located: bool = True

    @property
    def is_leaf(self) -> bool:
        return not self.children

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "section_num": self.section_num,
            "title": self.title,
            "depth": self.depth,
            "parent_id": self.parent_id,
            "children": list(self.children),
            "leaf": self.is_leaf,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "origin": self.origin,
            "word_count": self.word_count,
            "heading_located": self.heading_located,
        }


@dataclass
class ValidationIssue:
    code: str
    severity: str  # "error" blocks training, "warning" is reported only
    message: str
    node_id: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "node_id": self.node_id,
        }


@dataclass
class StructureResult:
    source_path: str
    source_sha256: str
    page_count: int
    nodes: list[TocNode]
    issues: list[ValidationIssue]
    unmapped_front_matter_words: int = 0
    book_title: str = ""

    @property
    def errors(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def passed(self) -> bool:
        return not self.errors

    @property
    def leaves(self) -> list[TocNode]:
        return [node for node in self.nodes if node.is_leaf]

    def node_by_id(self, node_id: str) -> TocNode:
        for node in self.nodes:
            if node.id == node_id:
                return node
        raise KeyError(node_id)

    def breadcrumb(self, node_id: str) -> str:
        parts: list[str] = []
        current: Optional[TocNode] = self.node_by_id(node_id)
        while current is not None:
            parts.append(current.title)
            current = self.node_by_id(current.parent_id) if current.parent_id else None
        return " > ".join(reversed(parts))
