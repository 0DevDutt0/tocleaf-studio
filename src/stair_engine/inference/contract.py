"""The training/inference contract: prompt text, ToC rendering, completion format and token layout.

Training and inference must see byte-identical prompts and token sequences, or the model is
evaluated on an input it was never trained on. Everything that affects that is defined here.

* Prompt text is read from the pinned upstream checkout (``src/stair/core/prompts.py``) with
  ``ast``. It is never copied into this repository, because the upstream repository has no
  licence. The SHA-256 of the strings is stored in every run manifest.
* The ToC text, completion format and token layout reproduce ``prepare_input_output_stair``
  and the TRL 1.14 prompt-completion tokenisation (``trl/trainer/sft_trainer.py``).
"""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PromptContract:
    system_prompt: str
    user_template: str
    source: str
    sha256: str

    def to_dict(self) -> dict:
        return {"source": self.source, "sha256": self.sha256}


def _module_string(source: str, name: str) -> str:
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            value = ast.literal_eval(node.value)  # implicit concatenation is folded by the parser
            if isinstance(value, str):
                return value
    raise KeyError(f"{name} is not a string assignment in the upstream prompts module")


def load_upstream_prompt_contract(upstream_root: str | Path) -> PromptContract:
    path = Path(upstream_root) / "src" / "stair" / "core" / "prompts.py"
    source = path.read_text(encoding="utf-8")
    system_prompt = _module_string(source, "SYSTEM_PROMPT")
    user_template = _module_string(source, "USER_PROMPT")
    digest = hashlib.sha256(json.dumps([system_prompt, user_template]).encode("utf-8")).hexdigest()
    return PromptContract(
        system_prompt=system_prompt,
        user_template=user_template,
        source=str(path),
        sha256=digest,
    )


@dataclass(frozen=True)
class ToCView:
    """A toc.json loaded the way the upstream loader and evaluator read it."""

    book_title: str
    toc_text: str  # every node, document order, as "section_num title"
    label_by_id: dict[str, str]
    id_by_label: dict[str, str]
    leaf_ids: frozenset[str]
    is_leaf_by_id: dict[str, bool]


def load_toc_view(toc_json_path: str | Path) -> ToCView:
    toc = json.loads(Path(toc_json_path).read_text(encoding="utf-8"))
    label_by_id: dict[str, str] = {}
    id_by_label: dict[str, str] = {}
    is_leaf_by_id: dict[str, bool] = {}
    lines: list[str] = []
    for node in toc["table_of_contents"]:
        label = f"{node['section_num']} {node['title']}"
        if label in id_by_label:
            raise ValueError(f"duplicate ToC label {label!r}; labels must be unique for constrained decoding")
        lines.append(label)
        label_by_id[node["id"]] = label
        id_by_label[label] = node["id"]
        is_leaf_by_id[node["id"]] = bool(node["leaf"])
    leaf_ids = frozenset(node_id for node_id, leaf in is_leaf_by_id.items() if leaf)
    return ToCView(
        book_title=toc["title"],
        toc_text="\n".join(lines),
        label_by_id=label_by_id,
        id_by_label=id_by_label,
        leaf_ids=leaf_ids,
        is_leaf_by_id=is_leaf_by_id,
    )


def render_messages(contract: PromptContract, view: ToCView, question: str) -> list[dict]:
    user = contract.user_template.format(title=view.book_title, toc=view.toc_text, question=question)
    return [
        {"role": "system", "content": contract.system_prompt},
        {"role": "user", "content": user},
    ]


def completion_text(labels: list[str]) -> str:
    """Target text, exactly as prepare_input_output_stair formats it."""

    quoted = [f'"{label}"' for label in labels]
    return "\n[" + ", ".join(quoted) + "]"


@dataclass(frozen=True)
class TokenLayout:
    """What the trainer fed the model for one example.

    ``prefix_ids`` is the input the model saw before the first completion token.
    ``target_ids`` are the completion tokens the loss was computed on, including the end-of-turn
    token. ``boundary_mismatch`` is True when the joint tokenisation does not start with the
    tokenised prompt, which is the case TRL reports with its "Mismatch between tokenized prompt"
    warning.
    """

    prefix_ids: list[int]
    target_ids: list[int]
    boundary_mismatch: bool


def training_token_layout(tokenizer, prompt_text: str, completion: str) -> TokenLayout:
    eos_text = tokenizer.eos_token
    completion_with_eos = completion if completion.endswith(eos_text) else completion + eos_text
    prompt_ids = tokenizer(prompt_text).input_ids
    joint_ids = tokenizer(prompt_text + completion_with_eos).input_ids
    boundary_mismatch = joint_ids[: len(prompt_ids)] != prompt_ids
    # TRL builds the completion mask from len(prompt_ids), so the trained input is the joint
    # prefix of that length and the trained targets are everything after it.
    return TokenLayout(
        prefix_ids=joint_ids[: len(prompt_ids)],
        target_ids=joint_ids[len(prompt_ids) :],
        boundary_mismatch=boundary_mismatch,
    )
