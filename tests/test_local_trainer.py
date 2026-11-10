"""The local trainer's loss and constrained scores, checked against Hugging Face's own loss and a manual computation.

A tiny randomly initialised Qwen2 model is used, so no download is needed. These tests catch off-by-one errors in the
completion positions, which would silently train on the wrong tokens.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from stair_engine.inference.constrained import TokenTrie  # noqa: E402
from stair_engine.training.local import TrainingExample, batch_nll, restricted_leaf_scores  # noqa: E402


def _tiny_model():
    torch.manual_seed(0)
    config = transformers.Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    return transformers.Qwen2ForCausalLM(config).eval()


def test_completion_loss_matches_hugging_face_causal_loss_on_a_padded_batch():
    model = _tiny_model()
    examples = [
        TrainingExample("a", prefix=[1, 2, 3, 4], target=[5, 6, 7], gold_leaf="x", question="q"),
        TrainingExample("b", prefix=[9, 8, 7], target=[1, 2], gold_leaf="y", question="q"),
    ]
    ours = batch_nll(model.model, model.lm_head, examples, pad_id=0, device="cpu")

    longest = max(len(e.prefix) + len(e.target) for e in examples)
    input_ids = torch.zeros((2, longest), dtype=torch.long)
    attention = torch.zeros((2, longest), dtype=torch.long)
    labels = torch.full((2, longest), -100, dtype=torch.long)
    for row, example in enumerate(examples):
        sequence = example.prefix + example.target
        input_ids[row, : len(sequence)] = torch.tensor(sequence)
        attention[row, : len(sequence)] = 1
        labels[row, len(example.prefix) : len(sequence)] = torch.tensor(example.target)
    reference = model(input_ids=input_ids, attention_mask=attention, labels=labels).loss

    assert torch.allclose(ours, reference, atol=1e-5), (float(ours), float(reference))


def test_constrained_scores_equal_a_manual_per_step_renormalisation():
    model = _tiny_model()
    sequences = {"a": [5, 6, 7], "b": [5, 9]}  # shared first token, then branches
    trie = TokenTrie(sequences)
    prefix = [1, 2, 3]
    scores = restricted_leaf_scores(model.model, model.lm_head, prefix, sequences, trie, "cpu")

    for leaf, target in sequences.items():
        with torch.no_grad():
            logits = model(input_ids=torch.tensor([prefix + target])).logits[0]
        expected = 0.0
        for step, token in enumerate(target):
            allowed = trie.allowed_next(tuple(target[:step]))
            step_logits = logits[len(prefix) - 1 + step][torch.tensor(allowed)]
            expected += float(torch.log_softmax(step_logits, dim=-1)[allowed.index(token)])
        assert scores[leaf] == pytest.approx(expected, abs=1e-4), leaf


def test_renormalised_scores_are_proper_probabilities_per_step():
    """Over the valid continuations at each step, the probabilities sum to one, so every leaf score is at most zero."""

    model = _tiny_model()
    sequences = {"a": [5, 6], "b": [5, 9], "c": [7]}
    trie = TokenTrie(sequences)
    scores = restricted_leaf_scores(model.model, model.lm_head, [1, 2], sequences, trie, "cpu")
    assert all(value <= 0.0 for value in scores.values())
