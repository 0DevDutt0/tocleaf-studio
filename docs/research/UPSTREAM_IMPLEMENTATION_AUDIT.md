# Upstream implementation audit

Upstream: `https://github.com/meghanadhpulivarthi/stair-clean`
Pinned commit: `3729c54f54e6e28aa3a4d714fc7ff8a3af72c0f3` (latest commit on `main` when this audit was made;
message: "fix: make vendored silt trainer work with the locked transformers 5.x / trl 1.x").

## 1. Licence: no permission to copy or redistribute

The repository has **no LICENSE file**, and the GitHub API reports `licenseInfo: null`. Without a
licence, the code is not available for copying into another repository or for redistribution.

Consequence (ADR-001): this project keeps the upstream in a separate checkout at the pinned commit
(`scripts/setup_upstream.sh`) and calls its CLI. It does not copy, vendor or redistribute upstream
source. Prompt text is read from the checkout at runtime and is never copied into this repository.

## 2. Environment, verified in WSL2 (Ubuntu) on an RTX 5090 Laptop GPU

- `uv sync` in the pinned checkout succeeded with the upstream `uv.lock`. Locked versions:
  torch 2.14.0 (CUDA 13.0 wheels from PyPI), transformers 5.17.0, trl 1.14.0, peft 0.21.1,
  accelerate 1.15.0, datasets 5.0.1, pypdfium2 5.13.0. The lock contains no PyTorch index, so the
  default PyPI wheels are used.
- A bf16 matmul on the GPU ran successfully (`cuda` available, capability (12, 0), torch arch list
  includes sm_120).
- `stair --help`, `stair train --help`, `stair eval --help`, `stair report-train --help`,
  `stair report-eval --help` and `stair prepare-data --help` all run. The arguments match the README.
- The upstream test suite passed: **122 passed in 28.86 s** (`pytest -q`, run in the upstream `.venv`).

Native Windows was not used for training. The lock resolves CUDA wheels for Linux, the vendored
trainer uses `torchrun`, and the project brief prefers WSL2 for execution.

## 3. Configuration (configs/base.yaml, verified)

| Area | Upstream default |
|---|---|
| Model | `Qwen/Qwen2.5-0.5B-Instruct` (not Mistral-7B; the paper's model is not the default) |
| LoRA | r = 16, alpha = 32, dropout 0.1, targets q/k/v/o/gate/down/up projections |
| Training | lr 1e-4, constant with warm-up (20 steps), 3 epochs, batch 4 x accumulation 4, max_seq_length 2048 |
| Early stopping | `eval_loss`, patience 5 evaluations; best checkpoint loaded at the end |
| Evaluation | ks [1, 3, 5, 10], greedy generation, max_new_tokens 128 |
| Data | QA pairs per chunk 3, chunk 2048 tokens (4 characters per token), split 80/10/10 random, seed 42 |

Override YAML files are deep-merged, and unknown keys are rejected. That behaviour was read in
`src/stair/config.py` and is what this project relies on.

## 4. Findings from reading the code

Each finding names the file and function it comes from.

### 4.1 No constrained decoding exists

A search for logits processors, `prefix_allowed_tokens`, tries, beam settings, grammars and
structured-output libraries found nothing. Evaluation (`src/stair/core/eval/inference.py`,
`src/stair/eval.py`) runs `model.generate(do_sample=False)`, then `ast.literal_eval`, then exact string
matching against the ToC lines. The paper's Stage 3 (constrained generation) is therefore **not in the
upstream code**. This project implements it separately (ADR-006).

### 4.2 Every ToC node is a leaf

`src/stair/core/data/toc.py::build_toc` sets `"leaf": true` on every section. Parent sections are
written to `toc.json` with the same flag. In `pdf_extract.py`, a bookmark's text runs up to the next
bookmark at any level, so parents with introductory text become sections and therefore targets.

### 4.3 Invalid output counts as valid when it names a parent

`eval.py` builds `title_to_id_map` from *all* ToC lines, including parents. An output that names a
non-leaf node is therefore "valid", so the upstream metrics overstate validity. Our harness classifies
these outputs as `non_leaf_identifier` and reports them separately.

### 4.4 The upstream metric called "hallucination_rate" is an invalid-identifier rate

`eval.py` computes `hallucination_rate = invalid entries / all entries`. Non-leaf outputs are not
counted in it (see 4.3). The paper defines hallucination as a non-leaf output. The two definitions do
not match, and this project uses the terms "invalid identifier rate" and "non-leaf identifier rate".

### 4.5 Splits are random over QA pairs

`src/stair/core/data/split.py` shuffles all QA pairs and cuts 80/10/10. It is not stratified by leaf,
so a leaf can end up with no training query. The paper's per-book splits are described in Section 3.2
of the paper, but not with a protocol this code can reproduce. This project stratifies by leaf (ADR-005).

### 4.6 A failed QA chunk is dropped silently

`qa_generation.py::generate_qa_pairs_for_chunk` retries three times, then returns `[]` after printing a
message. The caller continues without the chunk, and no report records the loss. This project records
every lost passage and fails the run when more than 20% are lost (`synthetic/qa.py`).

### 4.7 Checkpoint selection uses eval loss, not Recall@1

`metric_for_best_model: eval_loss`, and the paper uses dev Recall@1 with patience 20 epochs. The
upstream run is therefore not the paper's selection procedure. The paper's selection is available here
only through this project's own evaluation, which is not wired into training. This is a known gap.

### 4.8 Prompt and target format differ from the paper

Paper Table 2: a prompt that says "Only generate the section name", with `leaf` markers in the ToC and a
plain section name as the target. Upstream (`src/stair/core/prompts.py`): "Return the output only as a
Python list of strings", ToC lines without leaf markers, and a list-shaped target
(`\n["12.9 Step seven: ..."]`). This project uses the upstream format in the upstream-compatible profile,
and it reads the prompt strings from the pinned checkout so that training and inference match.

### 4.9 Token boundary mismatch (verified on this machine)

TRL 1.14 (`trl/trainer/sft_trainer.py`, prompt-completion branch) appends the EOS text to the completion,
tokenises the **joint** string `prompt + completion`, and checks that the tokenised prompt is a prefix of
it. It logs "Mismatch between tokenized prompt and the start of tokenized prompt+completion" when it is
not. For Qwen2.5-Instruct, the prompt ends with `assistant\n`, and the completion begins with `\n[`. The
two newlines merge into one token, so the prefix check fails.

Measured by `tocleaf preflight` on the fixture dataset: **101 of 101 examples mismatch**. Consequences:
the first completion token is not a training target, and upstream's greedy evaluation feeds the model a
prompt that ends with the unmerged newline, which differs from the training input. This project's
constrained decoder uses the training input exactly (`inference/contract.py::training_token_layout`).
The unconstrained evaluation reports both inputs, so the size of this effect can be measured.

### 4.10 CUDA only, and a silent skip

- `training_fsdp_trainer.py` calls `model.to("cuda")` unconditionally. There is no CPU path.
- `main()` returns without training when the run directory already contains a `checkpoint-*` directory,
  and the exit code is 0. This project creates a fresh run directory for every run and refuses to reuse one.
- Launching training from a shell that exits can kill the `torchrun` workers with SIGHUP (observed; see
  PROJECT_STATUS.md). Long runs must be started in a session that stays alive.

### 4.11 Other points

- `normalize_title` removes digit-only words from bookmark titles, but keeps "1.1" style numbers. Real
  books print numbers in titles, so numbering is handled differently here (ADR-004).
- The bundled example PDF (`data/example_book/sourdough_bread_guide.pdf`) has six flat bookmarks and
  metadata that says "untitled" and "anonymous". It cannot test a nested ToC, and its origin and licence
  are unknown, so this project does not use it for evaluation and does not redistribute it.
- `prepare-data` uses a default OpenAI client call with no provider-specific options. With a thinking model
  such as Qwen3.5 it returns empty content, because the output budget is spent on hidden reasoning (checked
  against the local endpoint). This project sends `reasoning_effort: "none"` for that reason.
- The paper extracted ToCs with PyMuPDF. The upstream uses pypdfium2 bookmarks, which is the same source
  data in a different library.

## 5. What this project takes from upstream, and how

| Upstream part | Use here | How |
|---|---|---|
| `stair train` (vendored trainer, LoRA, eval-loss early stopping) | used unmodified for the upstream-compatible training profile | subprocess via `uv run --project <checkout>`, own `.venv`, fresh run dir (`training/upstream.py`) |
| `toc.json`, `train/val/test.jsonl` schema | produced by our dataset builder with canonical ids and true leaf flags | `dataset/build.py` |
| Prompt text | read at runtime from the pinned checkout | `inference/contract.py` |
| Completion format | reproduced and checked against the upstream function in a test | `tests/test_retrieval_contract.py` (skipped when no checkout) |
| `prepare-data` (bookmarks, chunking, QA) | not used for the STAIR pipeline; used only for the baseline profile | our structure and QA modules replace it |
| `stair eval` | not used for results; it is a smoke test only | our harness reports every mode |

## 6. Open items

1. Recall@1 dev-based checkpoint selection is not wired into training.
2. The paper's split protocol cannot be reproduced from the public data (see the sourcing audit).
3. The effect of the boundary mismatch on accuracy is not yet measured on a real corpus.
