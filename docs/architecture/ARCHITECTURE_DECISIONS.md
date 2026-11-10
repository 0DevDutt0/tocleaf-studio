# Architecture decisions

Each record states the decision, the evidence behind it, and what would change it.

## ADR-001: Keep the upstream outside the repository

- **Decision.** The upstream `stair-clean` is cloned at the pinned commit into `~/stair-work/upstream/`
  by `scripts/setup_upstream.sh`. It is not copied, vendored or committed.
- **Evidence.** No LICENSE file; GitHub reports no licence (`licenseInfo: null`).
- **Changes if.** The upstream publishes a licence that permits redistribution, or the maintainer grants
  permission.

## ADR-002: Train and evaluate in WSL2; keep source on Windows

- **Decision.** Source and git live in `E:\Personal\STAIR`. Training, inference and tests run in WSL2 Ubuntu.
  Checkpoints, datasets and caches live on ext4 (`~/stair-work`, `~/.cache`).
- **Evidence.** The upstream lock has no PyTorch index, so Windows would receive a CPU-only torch wheel.
  CUDA is visible in WSL (`nvidia-smi`, bf16 kernel verified). The brief prefers WSL2 for execution.
- **Changes if.** A native Windows CUDA wheel is pinned and the trainer runs there without `torchrun` issues.

## ADR-003: Pin the same library versions as upstream, and isolate the environments

- **Decision.** This project pins torch 2.14.0, transformers 5.17.0, peft 0.21.1, accelerate 1.15.0,
  pypdfium2 5.13.0, openai 3.20.0, pyyaml 6.0.3 and numpy 2.5.3 (the upstream lock versions where they
  exist). Its virtualenv is `~/.venvs/tocleaf`. The upstream runs in its own `.venv`, which the adapter sets
  explicitly.
- **Evidence.** A shared `UPSTREAM` venv was observed to receive upstream packages through
  `uv run --project`, and it broke this project's environment (fixed by forcing the path; see PROJECT_STATUS.md).
  Adapter compatibility depends on the same versions.
- **Changes if.** The upstream pin changes (update both pins and re-run the contract tests).

## ADR-004: Canonical ids and printed numbers in the structure

- **Decision.** Nodes get stable ids (`n0000`...). A number printed in the bookmark title (`1.2.1 Field
  Capacity`, `Chapter 3: ...`) is the section number, and it must extend the parent's number and come after
  its previous sibling. Violations are errors. Without a printed number, the position-based number is used.
  Leaves are nodes with no children in the tree. Running headers and bare page numbers are removed only when
  repeated (and never when they match an outline title), and every removal is reported as a warning.
- **Evidence.** Real textbooks print numbers in titles. Upstream's leaf flag marks parents as leaves
  (`toc.py`), so the paper's definition has to be enforced here. A running header in every page would
  otherwise enter the section text.
- **Changes if.** A document's numbering is systematically unusable and a different scheme is needed.

## ADR-005: Split queries stratified by leaf

- **Decision.** Each leaf keeps at least one training query whenever it has more than one query. The split
  is seeded per leaf, so adding a leaf does not reshuffle the others. Duplicate normalised queries are removed
  before the split.
- **Evidence.** Upstream's random split over pooled QA pairs can leave a leaf with no training query, which
  STAIR cannot learn. The coverage gate (ADR-010) reports any such leaf.
- **Changes if.** The paper's exact per-book protocol becomes reproducible from its released data.

## ADR-006: Implement constrained decoding in this project

- **Decision.** The trie is built from the **exact target tokens** TRL used for training (`inference/contract.py`).
  The constrained result is the **exact ranking** of the valid leaves under constrained generation: each step is
  renormalised over the trie's valid continuations, computed by teacher forcing, so no beam pruning or length
  normalisation can change the order. Beam search with a prefix-restricting callable runs alongside as a generation
  check, with width equal to the leaf count and unnormalised scores. Violations are counted in an audit that must be zero.
- **Evidence.** Upstream has no constrained decoding (see the audit, section 4.1). The paper's Stage 3 says
  "restricting the output vocabulary to valid ToC leaf nodes" without an algorithm.
  Evidence from the fixture: a beam narrower than the leaf set, with length-normalised scores, gave R@1 0.590
  against an exact 0.769 on the same queries. The beam's top-1 agrees with the exact top-1 on 38 of 39 queries.
- **Limits.** Output is restricted to the trie, so a non-leaf or unknown label cannot be produced. Measured
  effect is reported separately from the unconstrained modes.
- **Changes if.** The paper or a later version specifies a different mechanism.

## ADR-007: The prompt contract comes from the pinned checkout

- **Decision.** `SYSTEM_PROMPT` and `USER_PROMPT` are read with `ast` from
  `src/stair/core/prompts.py` in the pinned checkout. Their SHA-256 is written to every run manifest.
  The completion string is reproduced and checked by a test that calls the upstream function itself.
- **Evidence.** Training and inference must see identical prompts. Copying the strings would make a licence
  question for upstream text, and reading them keeps one source of truth.
- **Changes if.** The upstream prompt changes (the hash changes, and the contract test fails until updated).

## ADR-008: Local model for synthetic queries, with thinking disabled

- **Decision.** Queries are generated through an OpenAI-compatible endpoint. The development endpoint is a local
  Ollama server with `qwen3.5:9b`, sent with `reasoning_effort: "none"`. Provenance records the model name,
  prompt version and timestamp.
- **Evidence.** The paper used Mixtral 8x7B, which is not available locally. A local thinking model returned empty
  content when reasoning was not disabled (`think: false` was ignored on the OpenAI-compatible route, and
  `reasoning_effort: "none"` worked). Generation parse failures are recovered by retry, and a tolerant field parser
  is used only when strict JSON fails (counted separately).
- **Changes if.** A stronger local model becomes available. The model used is part of the dataset fingerprint
  through `generation` in the manifest.

## ADR-009: Product name and licence

- **Decision.** The product name is **Tocleaf**, as in the repository README. The brief's "STAIR Studio" is kept
  as a description of the research engine only. The name lives in one constant (`branding.py`).
- **Licence.** MIT, copyright 0DevDutt0, decided by the user on 10 Nov 2026 ("anyone can use this repo"). The
  repository ships no upstream code and no benchmark data. The fixture PDF and the author-written queries are
  original and fall under the same licence. The upstream STAIR implementation has no licence of its own; the
  training path fetches it at setup, and the README says so.
- **Name.** The user confirmed "Tocleaf" on 10 Nov 2026 and asked for a suggestion only if there was a better
  option. None was found, so the name stays.
- **Changes if.** The user chooses a different product name or licence.

## ADR-010: Validation gates block training

- **Decision.** Structural errors stop dataset generation before any model call. The dataset gates refuse training
  when the structure failed, or when a leaf with text has no training query. The preflight refuses any example
  longer than `max_seq_length`, and reports the boundary-mismatch count.
- **Evidence.** Upstream trains silently on whatever the PDF produced, and silently drops failed QA chunks.
- **Changes if.** A user explicitly accepts a documented partial dataset (no such option exists yet, by design).

## ADR-011: Run the upstream trainer unmodified for the first real run

- **Decision.** The first real run uses the pinned trainer with overrides only (LoRA r16, alpha 32, the upstream
  model and prompts). The run directory is always fresh. Selection by Recall@1 on dev (the paper's method) is not
  wired in yet, and that gap is recorded as a deviation.
- **Evidence.** The trainer is CUDA-only and its selection metric is eval loss (audit sections 4.7 and 4.10).
  Modifying it would make the run less comparable to upstream and would need a second review.
- **Changes if.** Phase 2 needs paper-faithful selection. A patch would then be documented as a deviation, and
  the unpatched run kept for comparison.

## ADR-012: BM25 implemented in-house

- **Decision.** BM25 (k1 1.5, b 0.75) is implemented in `evaluation/bm25.py` over leaf sections (breadcrumb and
  section text), with no dependency.
- **Evidence.** The implementation is about 60 lines and easy to check. The paper's BM25 also uses leaves as units.
- **Changes if.** A standard library implementation is needed for a specific benchmark protocol.

## ADR-013: Local trainer that selects by the paper's rule

- **Decision.** A second training path (`stair_engine/training/local.py`) selects the checkpoint by constrained dev
  Recall@1, with early stopping on patience in epochs (paper, Section 5.2). The pinned upstream trainer remains the
  baseline, and it still selects by evaluation loss.
- **Evidence.** The upstream trainer runs a text-generation callback at every evaluation, which is slow and does not
  measure retrieval. The paper selects by dev Recall@1. The local loss matches Hugging Face's causal-LM loss on a
  padded batch, and its constrained scores match a manual per-step renormalisation (`tests/test_local_trainer.py`).
- **Deviations.** At most 20 epochs with patience 5 (paper: up to 200 epochs with patience 20). The dev split has
  about 18 queries, so the selection is noisy: one query is about 5 points of Recall@1. The optimiser and learning-rate
  schedule match the upstream configuration.

## ADR-014: Augmentation adds training queries only

- **Decision.** `augment-dataset` generates additional queries for the same passages and adds them to training. The
  validation and test files are copied byte for byte. A new query is rejected if it repeats an existing query, or if
  its content words overlap a validation or test query with a Jaccard score of at least 0.6.
- **Evidence.** A fixed test set makes runs comparable. The overlap check guards against paraphrases of test
  questions leaking into training. `tests/test_augmentation.py` checks both properties.
