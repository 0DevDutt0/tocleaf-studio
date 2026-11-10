"""Query generation: parsing, validation, provenance, failure budget; and dataset splits and gates."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stair_engine.dataset.build import SplitConfig, allocate_counts, build_dataset, stratified_split
from stair_engine.structure.build import analyze_pdf
from stair_engine.synthetic.qa import (
    generate_for_structure,
    parse_qa_items,
    split_into_passages,
    validate_candidate,
)

FIXTURE = Path(__file__).parent / "fixtures" / "irrigation_handbook.pdf"


class ScriptedClient:
    """A stand-in for the OpenAI-compatible client. It never touches the network."""

    model = "scripted-test-model"

    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = 0

    def complete(self, messages):
        self.calls += 1
        if not self._replies:
            return "[]"
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def _good_reply(question: str, answer: str = "An answer taken from the passage.") -> str:
    return json.dumps([{"question": question, "answer": answer, "type": "factual"}])


def test_strict_json_is_parsed():
    items = parse_qa_items('[{"question": "What is drainage?", "answer": "Removal of excess water.", "type": "definition"}]')
    assert items and items[0]["answer"] == "Removal of excess water."


def test_fenced_json_is_parsed():
    raw = '```json\n[{"question": "How deep is the root zone?", "answer": "About one metre.", "type": "factual"}]\n```'
    assert parse_qa_items(raw)[0]["type"] == "factual"


def test_almost_json_is_recovered_by_field_pairing():
    broken = (
        '[{"question": "When does the stage begin?", "answer": "At planting.",", "type": "factual"},'
        ' {"question": "What is field capacity?", "answer": "Water held after drainage.", "type": "definition"}]'
    )
    items = parse_qa_items(broken)
    assert [item["question"] for item in items] == ["When does the stage begin?", "What is field capacity?"]
    assert items[1]["answer"] == "Water held after drainage."


def test_non_answers_return_none():
    assert parse_qa_items("I cannot help with that.") is None


@pytest.mark.parametrize(
    ("question", "heading", "reason"),
    [
        ("Drainage?", "Drainage", "bad_length"),
        ("What is drainage in the passage?", "Drainage", "refers_to_passage"),
        ("What is drainage about?", "Drainage", "restates_heading"),
        ("How does poor drainage damage the roots of cereal crops?", "Drainage", None),
    ],
)
def test_candidate_validation(question, heading, reason):
    got_reason, _ = validate_candidate({"question": question, "answer": "x y z"}, heading)
    assert got_reason == reason


def test_passages_split_on_paragraphs_under_the_word_budget():
    text = "\n".join(["word " * 40] * 5)
    passages = split_into_passages(text, max_words=100)
    assert len(passages) == 3 and all(len(p.split()) <= 100 for p in passages)


def test_generation_records_provenance_and_rejections():
    structure = analyze_pdf(FIXTURE)
    leaves = [node for node in structure.leaves if node.id == structure.leaves[0].id]
    assert leaves
    reply = json.dumps(
        [
            {"question": "What is the permanent wilting point of a sandy loam?", "answer": "About 5 percent.", "type": "factual"},
            {"question": "What does the passage say about wilting?", "answer": "x", "type": "factual"},
        ]
    )
    client = ScriptedClient([reply] * 40)
    accepted, stats = generate_for_structure(structure, client, questions_per_passage=2)
    assert accepted, "at least one candidate should survive"
    assert all(q["reference"][0] in {leaf.id for leaf in structure.leaves} for q in accepted)
    assert all(q["provenance"]["model"] == "scripted-test-model" for q in accepted)
    assert stats.rejected["refers_to_passage"] >= 1


def test_generation_fails_loudly_when_the_model_is_unusable():
    structure = analyze_pdf(FIXTURE)
    client = ScriptedClient([RuntimeError("connection refused")] * 200)
    with pytest.raises(RuntimeError, match="no usable output"):
        generate_for_structure(structure, client, questions_per_passage=2, max_failure_fraction=0.2)


def test_allocation_keeps_a_training_query_for_every_leaf():
    config = SplitConfig(train=0.7, val=0.1, test=0.2)
    for n in range(1, 25):
        train, val, test = allocate_counts(n, config)
        assert train >= 1 and train + val + test == n, (n, train, val, test)


def test_stratified_split_is_deterministic_and_leaf_complete():
    records = []
    for leaf in ("n0001", "n0002", "n0003"):
        for index in range(6):
            records.append({"query_id": f"{leaf}-{index}", "question": f"q {leaf} {index}", "reference": [leaf]})
    first = stratified_split(records, SplitConfig(seed=7))
    second = stratified_split(records, SplitConfig(seed=7))
    assert first == second
    trained_leaves = {r["reference"][0] for r in first["train"]}
    assert trained_leaves == {"n0001", "n0002", "n0003"}
    all_ids = [r["query_id"] for split in first.values() for r in split]
    assert len(all_ids) == len(set(all_ids)) == len(records)


def test_build_dataset_writes_upstream_files_and_flags_uncovered_leaves(tmp_path):
    structure = analyze_pdf(FIXTURE)
    covered = structure.leaves[0]
    accepted = [
        {
            "question": f"Question number {i} about {covered.title.lower()}?",
            "answer": "answer",
            "reference": [covered.id],
            "question_type": "factual",
            "provenance": {"leaf_id": covered.id},
        }
        for i in range(10)
    ]
    manifest = build_dataset(structure, accepted, {"chunks": 1}, tmp_path, split=SplitConfig(seed=1))
    for name in ("toc.json", "train.jsonl", "val.jsonl", "test.jsonl", "docs.jsonl", "manifest.json", "structure.json"):
        assert (tmp_path / name).exists(), name
    toc = json.loads((tmp_path / "toc.json").read_text(encoding="utf-8"))
    assert {"id", "section_num", "title", "leaf"} <= set(toc["table_of_contents"][0])
    train_row = json.loads((tmp_path / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert {"question", "reference"} <= set(train_row) and train_row["reference"] == [covered.id]
    # Only one leaf is covered, so the coverage gate must report the others.
    assert manifest["coverage_passed"] is False
    assert len(manifest["uncovered_leaves"]) == len(structure.leaves) - 1


def test_build_dataset_rejects_queries_that_target_parents(tmp_path):
    structure = analyze_pdf(FIXTURE)
    parent = next(node for node in structure.nodes if not node.is_leaf)
    bad = [{"question": "Anything at all here?", "answer": "a", "reference": [parent.id], "question_type": "factual", "provenance": {}}]
    with pytest.raises(ValueError, match="non-leaf"):
        build_dataset(structure, bad, {}, tmp_path)
