"""Augmentation adds training queries and leaves the validation and test splits byte-for-byte unchanged."""

from __future__ import annotations

import json
from pathlib import Path

from stair_engine.dataset.build import SplitConfig, build_dataset
from stair_engine.dataset.preflight import check_dataset_gates
from stair_engine.pipeline import augment_dataset
from stair_engine.structure.build import analyze_pdf

FIXTURE = Path(__file__).parent / "fixtures" / "irrigation_handbook.pdf"


class ScriptedClient:
    """Returns one new query, one repeat of an existing test query, and one near-repeat, for every passage."""

    model = "scripted-test-model"

    def __init__(self, repeated: str, near: str):
        self.calls = 0
        self.repeated = repeated
        self.near = near

    def complete(self, messages):
        self.calls += 1
        return json.dumps(
            [
                {"question": f"Which idea from example {self.calls} matters most for the farm?", "answer": "a", "type": "factual"},
                {"question": self.repeated, "answer": "a", "type": "factual"},
                {"question": self.near, "answer": "a", "type": "factual"},
            ]
        )


def _base_dataset(tmp_path: Path) -> Path:
    structure = analyze_pdf(FIXTURE)
    accepted = []
    for leaf in structure.leaves:
        for index in range(6):
            accepted.append(
                {
                    "question": f"What does {leaf.title} explain about lesson {index}?",
                    "answer": "a",
                    "reference": [leaf.id],
                    "question_type": "factual",
                    "provenance": {"leaf_id": leaf.id},
                }
            )
    base = tmp_path / "base"
    build_dataset(structure, accepted, {}, base, split=SplitConfig(train=0.5, val=0.2, test=0.3, seed=1))
    return base


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_augmentation_adds_training_queries_and_keeps_eval_splits_unchanged(tmp_path):
    base = _base_dataset(tmp_path)
    test_rows = _rows(base / "test.jsonl")
    repeated = test_rows[0]["question"]
    near = repeated + " today"
    client = ScriptedClient(repeated=repeated, near=near)

    out = tmp_path / "augmented"
    manifest = augment_dataset(base, FIXTURE, out, client=client, questions_per_passage=3, max_passage_words=1500)

    # Validation and test are byte-for-byte unchanged.
    assert (out / "test.jsonl").read_bytes() == (base / "test.jsonl").read_bytes()
    assert (out / "val.jsonl").read_bytes() == (base / "val.jsonl").read_bytes()

    base_train = _rows(base / "train.jsonl")
    out_train = _rows(out / "train.jsonl")
    added = manifest["augmentation"]["added_train_queries"]
    assert added >= 1
    assert len(out_train) == len(base_train) + added
    assert len({row["query_id"] for row in out_train}) == len(out_train)

    rejected = manifest["augmentation"]["rejected"]
    assert rejected["duplicate_of_existing_query"] >= 1
    assert rejected["near_duplicate_of_validation_or_test"] >= 1
    assert manifest["coverage_passed"] is True
    assert manifest["augmentation"]["validation_and_test_unchanged"] is True
    assert check_dataset_gates(out)["fingerprint_sha256"] == manifest["fingerprint_sha256"]


def test_augmentation_refuses_a_structure_that_differs_from_the_base(tmp_path):
    import pytest

    base = _base_dataset(tmp_path)
    corrections_path = tmp_path / "corrections.json"
    corrections_path.write_text(
        json.dumps({"exclude": [{"title": "Drainage", "page": 1, "reason": "test: changes the ToC"}]}),
        encoding="utf-8",
    )
    from stair_engine.structure.overrides import load_corrections

    with pytest.raises(ValueError, match="differs from the base dataset"):
        augment_dataset(
            base,
            FIXTURE,
            tmp_path / "out",
            client=ScriptedClient(repeated="x", near="y"),
            corrections=load_corrections(corrections_path),
        )
