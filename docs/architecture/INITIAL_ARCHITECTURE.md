# Initial architecture

Status: describes the code that exists in this repository after the Phase 1 vertical slice. Modules
marked *planned* do not exist yet.

## 1. Two runtimes, one contract

| Runtime | What runs there | Why |
|---|---|---|
| Windows 11 (`E:\Personal\STAIR`) | source, documents, git, the editor | the project folder and repository live here |
| WSL2 Ubuntu (`/mnt/e/Personal/STAIR` for source; `~/stair-work` on ext4 for data) | training, inference, tests, the pinned upstream | the upstream lock resolves CUDA wheels for Linux; `torchrun` and the trainer are Linux-tested; checkpoints stay on ext4 |

Two Python environments exist in WSL and must never share a directory:

- `~/.venvs/tocleaf`: this project (`UV_PROJECT_ENVIRONMENT`, set by `scripts/setup_env.sh`).
- `~/stair-work/upstream/stair-clean/.venv`: the pinned upstream. The adapter forces this path, because
  an inherited `UV_PROJECT_ENVIRONMENT` previously made uv install upstream packages into this project's
  venv (observed, fixed; see PROJECT_STATUS.md).

The upstream checkout is outside the repository (ADR-001).

## 2. Pipeline

```
PDF ──► structure ──► validation report ──(errors block)──►
        │ (outline, text spans, page ranges, canonical ids)
        ▼
synthetic queries (OpenAI-compatible endpoint) ──► validation, dedupe, provenance
        │
        ▼
dataset (toc.json, train/val/test.jsonl, docs.jsonl, manifest with fingerprint and coverage)
        │
        ├── gates: structure passed, every leaf has a training query
        ├── preflight: token length of every example, boundary-mismatch count
        ▼
upstream trainer (`stair train`, unmodified, LoRA, fresh run directory)
        │
        ▼
run directory: checkpoint-best (adapter), config.json, train.log, gpu_memory.csv, run_manifest.json
        │
        ▼
evaluation: constrained beam search and two unconstrained modes; BM25 baseline; metrics by failure type
```

## 3. Modules

```
src/stair_engine/
  branding.py                 product name (single constant: "Tocleaf")
  structure/
    models.py                 TocNode, ValidationIssue, StructureResult (canonical ids, leaf = no children)
    tree.py                   outline -> tree; printed numbers; hierarchy and order checks
    pdf_reader.py             pypdfium2 outline and text; running-text removal; heading location
    build.py                  analysis orchestration; per-node text spans; upstream toc.json shape
  synthetic/
    llm.py                    OpenAI-compatible client; env-only configuration; retry classification
    qa.py                     prompt, strict and tolerant parsing, validation, provenance, failure budget
  dataset/
    build.py                  dataset writer, stratified split, coverage gate, manifest and fingerprint
    preflight.py              token-length check (refuses overflow); dataset gates before training
  inference/
    contract.py               prompt text read from pinned upstream; ToC view; completion format;
                              token layout as TRL builds it (training input prefix and targets)
    constrained.py            token trie over leaf completions; prefix-restricting callable; audit
  evaluation/
    metrics.py                recall, hit, MRR, nDCG; output classification; per-failure rates
    bm25.py                   dependency-free BM25 over leaf sections
    stair_runner.py           loads adapter; runs three decoding modes; writes per-query and summary files
  training/
    upstream.py               runs the pinned trainer; GPU sampling; run manifest
  pipeline.py                 end-to-end steps with the validation gate
  cli.py                      `tocleaf analyze | build-dataset | preflight | train | eval-stair | eval-bm25`
```

Planned, not built: a FastAPI service (`backend/`), SQLite metadata, job records, the React frontend,
hybrid retrieval (passages, rank fusion, reranking), grounded answers with citation checks, and
multi-document routing. These follow the phase gates in `IMPLEMENTATION_PLAN.md`.

## 4. Data contracts

- **Canonical identifier.** Every ToC node has an id (`n0000`, `n0001`, ...) in `structure.json` and
  `toc.json`. Training targets and evaluation use the id. The model's text output is mapped to an id
  only through the exact `section_num title` label, and labels must be unique (checked).
- **toc.json** keeps the upstream keys `title`, `table_of_contents[].id|section_num|title|leaf`. Extra keys
  (`parent_id`, `depth`, page range) are ignored by the upstream loader.
- **train/val/test.jsonl** records carry `question` and `reference` (a list with one leaf id), plus
  `query_id`, `answer` and `question_type`. Provenance stays in `qa_candidates.jsonl`, so the trainer's
  metadata stays small.
- **manifest.json** carries the dataset fingerprint (SHA-256 of ToC and splits), the split configuration,
  per-leaf coverage, generator statistics (model, prompt version, parse failures, tolerant parses,
  rejections by reason, lost passages), and the completion-format description.
- **run_manifest.json** carries the upstream commit, the exact command, overrides, dataset fingerprint,
  prompt-contract SHA-256, return code, peak GPU memory, checkpoints, and the boundary-mismatch warning count.

## 5. Decoding modes (evaluation)

| Mode | Input tokens | Restriction | Purpose |
|---|---|---|---|
| `constrained` | training prefix | beam search over the leaf trie; top-K distinct leaves | paper's Stage 3 |
| `unconstrained` | training prefix | greedy | the trained model without a restriction |
| `unconstrained_upstream_prompt` | prompt as upstream `eval` tokenises it | greedy | measures the boundary-token effect (contract.py) |

Every output is classified as `valid_leaf`, `non_leaf`, `invalid_identifier`, or `unparseable`. Only
`valid_leaf` outputs contribute to the ranking.

## 6. Storage

Workspace (`~/stair-work/workspace/<document>/`): `validation.json`, `dataset/` (see contracts).
Runs (`~/stair-work/runs/<run-id>/`): training outputs; never reused (the upstream would skip training).
Evaluations (`~/stair-work/evals/<run-id>/`, planned location): per-query JSONL and summaries.
Nothing large is committed. `.gitignore` excludes `runs/`, `data/generated/`, checkpoints and the
upstream `.venv`.

## 7. Hardware observations that shape the design

- The RTX 5090 Laptop GPU reports 24,463 MiB. Ollama's runner held about 11.5 GB of it at idle during this
  work, so training budgets about 12 GB. The peak training usage is recorded per run in `gpu_memory.csv` and
  `run_manifest.json` (`peak_gpu_memory_mib`); the measured value is reported in PROJECT_STATUS.md.
- A 0.5B model fits the 2048-token limit with room to spare for the ToCs seen so far (140 tokens for 19 nodes).
  The budget must be re-checked per document, which is why preflight refuses overflow.
