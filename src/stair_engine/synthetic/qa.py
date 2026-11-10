"""Generate synthetic queries for leaf sections, with validation and provenance.

Every accepted query is tied to exactly one canonical leaf id. Parent nodes never become
training targets. Rejected candidates are counted by reason, so the generation report shows
how much of the model's output was usable.
"""

from __future__ import annotations

import ast
import datetime as dt
import json
import re
from collections import Counter
from dataclasses import dataclass, field


from stair_engine.structure.models import StructureResult, TocNode
from stair_engine.synthetic.llm import ChatClient

PROMPT_VERSION = "tocleaf-qa-v1"
MAX_PARSE_ATTEMPTS = 3

QUESTION_TYPES = (
    "factual",
    "definition",
    "conceptual",
    "explanatory",
    "application",
    "relationship",
    "paraphrase",
    "conversational",
)

STOPWORDS = frozenset(
    """a an and are as at be by can do does for from how i in into is it its me my of on or
    our should so tell that the their them there these this to was we what when where which
    who why will with would you your explain describe define give list about between under
    over across text passage section chapter book""".split()
)
_PASSAGE_LEAK = re.compile(r"\b(passage|the text|the context|this section|the excerpt)\b", re.IGNORECASE)

SYSTEM_PROMPT = (
    "You write the questions a reader would type into a search box to find a specific passage in a book. "
    "You answer only from the passage you are given. You reply with a JSON array and nothing else."
)


def build_messages(book_title: str, breadcrumb: str, passage: str, questions_per_passage: int) -> list[dict]:
    types = ", ".join(QUESTION_TYPES)
    user_prompt = (
        f"Book: {book_title}\n"
        f"Location in the book: {breadcrumb}\n\n"
        "Passage:\n"
        f"<<<\n{passage}\n>>>\n\n"
        f"Write {questions_per_passage} questions that this passage answers. Spread them over different question "
        f"types from this list where possible: {types}.\n"
        "Rules:\n"
        "- Each question must be answerable from the passage alone and must make sense without it.\n"
        "- Do not copy the location heading, and do not repeat one question in different words.\n"
        "- Prefer natural wording and synonyms to the exact terms of the heading.\n"
        "- Never mention 'the passage', 'the text' or 'the section' in a question.\n"
        "- Each answer is one or two sentences taken from the passage.\n\n"
        'Return only a JSON array like [{"question": "...", "answer": "...", "type": "factual"}].'
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _strip_code_fence(raw: str) -> str:
    text = raw.strip()
    fence = re.match(r"^```[a-zA-Z]*\s*(.*?)\s*```$", text, re.DOTALL)
    return fence.group(1) if fence else text


_STRING_BODY = r'((?:[^"\\]|\\.)*)'
_QUESTION_FIELD = re.compile(r'"question"\s*:\s*"' + _STRING_BODY + '"', re.DOTALL)
_ANSWER_FIELD = re.compile(r'"answer"\s*:\s*"' + _STRING_BODY + '"', re.DOTALL)
_TYPE_FIELD = re.compile(r'"type"\s*:\s*"([^"]*)"')


def _tolerant_items(text: str) -> list[dict] | None:
    """Recover question/answer/type triples when the reply is almost, but not quite, valid JSON.

    Small models often close a string early and leave a stray quote. The fields are still in
    order, so they are paired by position. The result is flagged by the caller as tolerant.
    """

    questions = list(_QUESTION_FIELD.finditer(text))
    if not questions:
        return None
    answers = list(_ANSWER_FIELD.finditer(text))
    types = list(_TYPE_FIELD.finditer(text))
    items: list[dict] = []
    for index, question in enumerate(questions):
        next_question = questions[index + 1].start() if index + 1 < len(questions) else len(text)
        answer = next((a for a in answers if question.end() <= a.start() < next_question), None)
        query_type = next((t for t in types if question.end() <= t.start() < next_question), None)
        if answer is None:
            continue
        items.append(
            {
                "question": question.group(1).replace('\\"', '"').strip(),
                "answer": answer.group(1).replace('\\"', '"').strip(),
                "type": query_type.group(1) if query_type else "",
            }
        )
    return items or None


def _strict_items(raw: str) -> list[dict] | None:
    text = _strip_code_fence(raw)
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end <= start:
        return None
    candidate = text[start : end + 1]
    parsed: object
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(candidate)
        except (ValueError, SyntaxError):
            return None
    if not isinstance(parsed, list):
        return None
    return [item for item in parsed if isinstance(item, dict)]


def parse_qa_items(raw: str) -> list[dict] | None:
    """Return the candidate dicts, or None. Strict JSON first, then a tolerant pairing of fields."""

    strict = _strict_items(raw)
    if strict is not None:
        return strict
    return _tolerant_items(_strip_code_fence(raw))


def normalise_question(question: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", question.lower()).strip()


def _content_tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if token not in STOPWORDS}


def validate_candidate(item: dict, heading: str) -> tuple[str | None, str | None]:
    """Return (reason, None) when the candidate is rejected, or (None, normalised_type) otherwise."""

    question = item.get("question")
    answer = item.get("answer")
    if not isinstance(question, str) or not isinstance(answer, str):
        return "missing_fields", None
    question = question.strip()
    if len(question.split()) < 4 or len(question.split()) > 45:
        return "bad_length", None
    if not answer.strip():
        return "empty_answer", None
    if _PASSAGE_LEAK.search(question):
        return "refers_to_passage", None
    if normalise_question(question) == normalise_question(heading):
        return "copies_heading", None
    if _content_tokens(question) and _content_tokens(question) <= _content_tokens(heading):
        return "restates_heading", None
    query_type = str(item.get("type", "")).strip().lower()
    return None, query_type if query_type in QUESTION_TYPES else "unspecified"


@dataclass
class GenerationStats:
    chunks: int = 0
    attempts: int = 0
    parse_failures: int = 0
    tolerant_parses: int = 0
    request_failures: int = 0
    candidates: int = 0
    accepted: int = 0
    rejected: Counter = field(default_factory=Counter)
    duplicates: int = 0
    failed_chunks: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "chunks": self.chunks,
            "attempts": self.attempts,
            "parse_failures": self.parse_failures,
            "parse_failure_rate": (self.parse_failures / self.attempts) if self.attempts else 0.0,
            "tolerant_parses": self.tolerant_parses,
            "request_failures": self.request_failures,
            "candidates": self.candidates,
            "accepted": self.accepted,
            "rejected_by_reason": dict(self.rejected),
            "duplicates_removed": self.duplicates,
            "failed_chunks": self.failed_chunks,
        }


def split_into_passages(text: str, max_words: int) -> list[str]:
    """Split a leaf's text on paragraph boundaries into passages of at most max_words words."""

    paragraphs = [paragraph.strip() for paragraph in text.split("\n") if paragraph.strip()]
    passages: list[str] = []
    current: list[str] = []
    current_words = 0
    for paragraph in paragraphs:
        words = len(paragraph.split())
        if current and current_words + words > max_words:
            passages.append("\n".join(current))
            current, current_words = [], 0
        current.append(paragraph)
        current_words += words
    if current:
        passages.append("\n".join(current))
    return passages


def generate_for_structure(
    structure: StructureResult,
    client: ChatClient,
    *,
    questions_per_passage: int = 4,
    max_passage_words: int = 900,
    min_words_in_section: int = 5,
    max_failure_fraction: float = 0.2,
) -> tuple[list[dict], GenerationStats]:
    """Generate and validate queries for every leaf that has enough text.

    The run stops early and raises when more than ``max_failure_fraction`` of the attempted
    passages failed to produce usable output, so a broken model cannot silently yield a
    thin dataset.
    """

    stats = GenerationStats()
    accepted: list[dict] = []
    seen_questions: set[str] = set()
    timestamp = dt.datetime.now(dt.timezone.utc).isoformat()

    leaves: list[TocNode] = [node for node in structure.nodes if node.is_leaf and node.word_count >= min_words_in_section]
    for leaf in leaves:
        heading = f"{leaf.section_num} {leaf.title}"
        breadcrumb = structure.breadcrumb(leaf.id)
        for passage_index, passage in enumerate(split_into_passages(leaf.text, max_passage_words)):
            stats.chunks += 1
            messages = build_messages(structure.book_title, breadcrumb, passage, questions_per_passage)
            items: list[dict] | None = None
            last_raw = ""
            for _attempt in range(1, MAX_PARSE_ATTEMPTS + 1):
                stats.attempts += 1
                try:
                    last_raw = client.complete(messages)
                except Exception as error:  # noqa: BLE001 - recorded, then counted against the failure budget
                    stats.request_failures += 1
                    stats.failed_chunks.append(
                        {"leaf_id": leaf.id, "passage": passage_index, "reason": "request_error", "error": str(error)[:300]}
                    )
                    break
                items = parse_qa_items(last_raw)
                if items is not None:
                    if _strict_items(last_raw) is None:
                        stats.tolerant_parses += 1
                    break
                stats.parse_failures += 1
            if items is None:
                if not any(entry["leaf_id"] == leaf.id and entry["passage"] == passage_index for entry in stats.failed_chunks):
                    stats.failed_chunks.append(
                        {
                            "leaf_id": leaf.id,
                            "passage": passage_index,
                            "reason": "unparseable_after_retries",
                            "raw_excerpt": (last_raw[:200] + " ... " + last_raw[-200:]) if len(last_raw) > 400 else last_raw,
                        }
                    )
                continue

            for item in items:
                stats.candidates += 1
                reason, query_type = validate_candidate(item, heading)
                if reason is not None:
                    stats.rejected[reason] += 1
                    continue
                question = str(item["question"]).strip()
                key = normalise_question(question)
                if key in seen_questions:
                    stats.duplicates += 1
                    continue
                seen_questions.add(key)
                stats.accepted += 1
                accepted.append(
                    {
                        "question": question,
                        "answer": str(item["answer"]).strip(),
                        "reference": [leaf.id],
                        "question_type": query_type,
                        "provenance": {
                            "leaf_id": leaf.id,
                            "section_num": leaf.section_num,
                            "heading": leaf.title,
                            "passage_index": passage_index,
                            "model": client.model,
                            "prompt_version": PROMPT_VERSION,
                            "generated_at": timestamp,
                        },
                    }
                )

    lost_passages = len({(entry["leaf_id"], entry["passage"]) for entry in stats.failed_chunks})
    if stats.chunks and lost_passages / stats.chunks > max_failure_fraction:
        raise RuntimeError(
            f"Question generation produced no usable output for {lost_passages} of {stats.chunks} passages "
            f"(limit {max_failure_fraction:.0%}). Check the endpoint, model and prompt before retrying."
        )
    return accepted, stats
