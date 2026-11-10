"""Constrained generation over the valid leaf completions.

The paper's inference step restricts output to valid ToC leaves. That is implemented here as a
token trie built from the exact completion tokens the trainer used (see contract.py). At every
step only tokens that continue a valid leaf completion are allowed. The output therefore cannot
name a non-leaf or an unknown section, and the evaluation still checks this on every query.

Output validation and constrained decoding are different steps. Validation checks text after
generation. Decoding restricts the model's choices during generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

LEAF_MARK = "__leaf__"


class TokenTrie:
    """A prefix tree over token-id sequences, each ending in the end-of-turn token."""

    def __init__(self, sequences_by_leaf: dict[str, list[int]]):
        if not sequences_by_leaf:
            raise ValueError("a trie needs at least one leaf sequence")
        self._root: dict = {}
        self._sequences = {leaf: tuple(seq) for leaf, seq in sequences_by_leaf.items()}
        for leaf_id, sequence in self._sequences.items():
            if not sequence:
                raise ValueError(f"leaf {leaf_id} has an empty completion")
            node = self._root
            for token in sequence:
                node = node.setdefault(token, {})
            if LEAF_MARK in node:
                raise ValueError(f"two leaves share the completion of {leaf_id}")
            node[LEAF_MARK] = leaf_id
        self._check_prefix_free()

    def _check_prefix_free(self) -> None:
        # A completion that is a strict prefix of another would make the end of one unreachable.
        for leaf_id, sequence in self._sequences.items():
            for other_id, other in self._sequences.items():
                if other_id != leaf_id and len(other) > len(sequence) and other[: len(sequence)] == sequence:
                    raise ValueError(f"completion of {leaf_id} is a prefix of {other_id}")

    def allowed_next(self, prefix: tuple[int, ...] | list[int]) -> list[int]:
        node = self._root
        for token in prefix:
            node = node.get(token)
            if node is None:
                return []
        return [token for token in node if token != LEAF_MARK]

    def leaf_for(self, sequence: tuple[int, ...] | list[int]) -> str | None:
        node = self._root
        for token in sequence:
            node = node.get(token)
            if node is None:
                return None
        return node.get(LEAF_MARK)

    @property
    def leaf_count(self) -> int:
        return len(self._sequences)


@dataclass
class ConstraintAudit:
    """Counts constraint violations, which must stay at zero for a valid constrained run."""

    beams_checked: int = 0
    unknown_prefix_steps: int = 0
    sequences_outside_trie: int = 0
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "beams_checked": self.beams_checked,
            "unknown_prefix_steps": self.unknown_prefix_steps,
            "sequences_outside_trie": self.sequences_outside_trie,
        }


def make_prefix_allowed_tokens_fn(trie: TokenTrie, prefix_length: int, eos_token_id: int, audit: ConstraintAudit):
    """Build the callable that transformers' beam search uses to restrict the next token."""

    def allowed(_batch_id: int, input_ids) -> list[int]:
        generated = tuple(int(token) for token in input_ids[prefix_length:].tolist())
        candidates = trie.allowed_next(generated)
        if not candidates:
            # A complete leaf is a finished hypothesis, and its only legal continuation is end-of-turn.
            # Anything else here means the model left the trie, which the audit must report.
            if trie.leaf_for(generated) is None:
                audit.unknown_prefix_steps += 1
            return [eos_token_id]
        return candidates

    return allowed


def truncate_at_eos(tokens: list[int], eos_token_id: int) -> list[int]:
    if eos_token_id in tokens:
        return tokens[: tokens.index(eos_token_id) + 1]
    return tokens


def sequences_to_leaves(
    generated_sequences: list[list[int]],
    trie: TokenTrie,
    eos_token_id: int,
    audit: ConstraintAudit,
) -> list[str | None]:
    leaves: list[str | None] = []
    for tokens in generated_sequences:
        audit.beams_checked += 1
        leaf = trie.leaf_for(tuple(truncate_at_eos(tokens, eos_token_id)))
        if leaf is None:
            audit.sequences_outside_trie += 1
        leaves.append(leaf)
    return leaves
