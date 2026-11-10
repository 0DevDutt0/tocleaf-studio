"""Token-level regression tests against the real Qwen2.5-Instruct tokenizer (CPU only).

These guard the training/inference contract. If the upstream tokenisation changes, or a prompt edit moves
the boundary, these tests fail before any GPU time is spent.
"""

from __future__ import annotations

import pytest

from stair_engine.inference.contract import completion_text, training_token_layout

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(MODEL)
    except OSError as error:  # no network and no cached tokenizer
        pytest.skip(f"tokenizer unavailable offline: {error}")


def _prompt(tokenizer, question: str) -> str:
    messages = [
        {"role": "system", "content": "You select sections."},
        {"role": "user", "content": f"Table of Contents:\n1.1 Alpha\n1.2 Beta\n\nQuery:\n{question}\n\nRelevant Sections:"},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def test_target_tokens_do_not_depend_on_the_question(tokenizer):
    short = training_token_layout(tokenizer, _prompt(tokenizer, "Why?"), completion_text(["1.2 Beta"]))
    long = training_token_layout(
        tokenizer,
        _prompt(tokenizer, "What does the text say about soil moisture under drought conditions in sandy fields?"),
        completion_text(["1.2 Beta"]),
    )
    assert short.target_ids == long.target_ids
    assert len(short.prefix_ids) != len(long.prefix_ids)  # the prefix length is per query, as the code assumes


def test_boundary_newline_merges_with_the_completion(tokenizer):
    prompt = _prompt(tokenizer, "What is beta?")
    layout = training_token_layout(tokenizer, prompt, completion_text(["1.2 Beta"]))
    assert layout.boundary_mismatch is True
    # The trained input is the tokenised prompt with its trailing newline (the completion's first newline
    # merges into it), so the prefix equals the tokenisation of prompt + "\n".
    assert layout.prefix_ids == tokenizer(prompt + "\n").input_ids
    assert layout.target_ids[-1] == tokenizer.eos_token_id


def test_targets_decode_back_to_the_leaf_label(tokenizer):
    layout = training_token_layout(tokenizer, _prompt(tokenizer, "Anything?"), completion_text(["1.2 Beta"]))
    text = tokenizer.decode(layout.target_ids, skip_special_tokens=True)
    assert text == '["1.2 Beta"]'
