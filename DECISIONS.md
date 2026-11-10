# Decisions log

Architectural decisions with their evidence are in `docs/architecture/ARCHITECTURE_DECISIONS.md` (ADR-001 to
ADR-012). This file records the decisions taken while executing the work, in date order, with what changed
because of them.

| Date | Decision | Why | Reversal condition |
|---|---|---|---|
| 10 Nov 2026 | Project folder is the repository root; `origin` is `0DevDutt0/tocleaf-studio`; no commit or push | the working folder was not a git repo; the user did not ask for commits | the user asks for a commit |
| 10 Nov 2026 | `STAIR/` (the existing venv) is ignored and left untouched | created by the user; it is only a bare venv | the user asks to remove or reuse it |
| 10 Nov 2026 | Product name "Tocleaf" (repository README), brief's "STAIR Studio" kept as the research engine's description | the repository README is the user's most recent text | the user names a different product |
| 10 Nov 2026 | Upstream kept in `~/stair-work/upstream/stair-clean` at `3729c54`, not in the repository | no licence on upstream (ADR-001) | licence granted |
| 10 Nov 2026 | Training and evaluation in WSL2; source on Windows | CUDA wheels resolve for Linux; WSL sees the GPU (ADR-002) | a verified native Windows path |
| 10 Nov 2026 | Our environment pins upstream's versions; the upstream runs in its own `.venv` | a shared venv was corrupted by `uv run --project` (see PROJECT_STATUS.md) | upstream pin changes |
| 10 Nov 2026 | Local Ollama `qwen3.5:9b` generates queries with `reasoning_effort: "none"` | Mixtral is not available; thinking consumed the output budget (ADR-008) | a stronger local model |
| 10 Nov 2026 | Fixture document is our own text (`make_fixture_pdf.py`) | the upstream sample has no licence and cannot test nested ToCs | a real open textbook is selected for Phase 1 |
| 10 Nov 2026 | Split stratified by leaf, 0.7/0.1/0.2 for the fixture, 0.8/0.1/0.1 default | every leaf must be trainable (ADR-005) | paper protocol becomes reproducible |
| 10 Nov 2026 | Training overrides for the fixture: 20 epochs, batch 4, accumulation 2, evaluation every 90 steps (changed from 8, then 30, because the generation callback made each evaluation take minutes), patience 6, lr 1e-4 (upstream default), LoRA r16 alpha 32 | the upstream default of 3 epochs with evaluation every 30 steps would evaluate about once for 101 queries | a larger corpus is used; then use upstream defaults |
| 10 Nov 2026 | Constrained decoding uses the trained input prefix and the TRL target tokens | upstream has none; the boundary merge changes the input (audit 4.9) | the trainer's tokenisation changes |
| 10 Nov 2026 | Running-text removal is limited to repeated short lines and bare numbers, and is reported | real books carry headers and footers that would enter section text | a document that needs a different rule |
| 10 Nov 2026 | Printed section numbers are authoritative; mismatches are errors | the book's own numbering is what users see (ADR-004) | a book where printed numbers are unreliable |
| 10 Nov 2026 | Upstream trainer run without modification for the first real run | comparability and review burden (ADR-011) | Phase 2 needs paper-faithful selection |
| 10 Nov 2026 | Constrained result is the exact per-step renormalised ranking over valid leaves; the beam is a generation check with width equal to the leaf count and unnormalised scores | the first beam (width 10, length-normalised) gave R@1 0.590 against an exact 0.769; see PROJECT_STATUS.md, issue 13 | the paper specifies a different decoding method |
| 10 Nov 2026 | The author-written query set is reported separately from the synthetic split | the synthetic split is saturated (BM25 R@1 1.000) and cannot separate methods | a larger, independently written query set exists |
| 10 Nov 2026 | Code was pushed to branch `phase-1-vertical-slice`, not to `main`, and no pull request was opened | default-branch guidance; the user asked for the files to be pushed, not merged | the user asks for a merge or a pull request |
| 10 Nov 2026 | Fixture training was restarted as r8 with the same settings as r7 | the user asked for the restart after the reaper stopped r7 at step 147 | any change to the training settings |
| 10 Nov 2026 | Licence: MIT, copyright 0DevDutt0 (LICENSE; README section updated) | the user asked for anyone to be able to use the repository | the user wants an explicit patent grant (then Apache-2.0) |
| 10 Nov 2026 | Product name stays "Tocleaf" (the repository name and README) | the user asked to follow the repository name unless a better suggestion exists; none was found | the user chooses another name |
| 10 Nov 2026 | Headline comparison fixed before its results: local trainer, the paper's rule (dev constrained R@1, patience 5), constrained beam R@1 on the same 47 test queries; every run reported | choosing the headline after seeing the numbers would bias the comparison (advisor review) | an independent query set exists |
| 10 Nov 2026 | Dev-loss control added to the local trainer: the same run also keeps and evaluates the lowest-dev-loss adapter (`checkpoint-best-loss`) | isolates the selection rule while trainer, data and tokenisation stay fixed | never; it is a control |
| 10 Nov 2026 | Upstream run on augmented data keeps the earlier run's schedule; only the data and the evaluation interval change (one epoch, 85 steps) | isolates the data effect with the same trainer | the earlier run's schedule changes |
| 10 Nov 2026 | Correction: the upstream trainer needs about 2.9 GB of GPU memory, not 14.4 GB; the earlier peak included Ollama's idle runner (11.5 GB) | the earlier run's GPU samples begin at 11,522 MiB, before training starts | another model or batch size is used |
| 10 Nov 2026 | Stopping Ollama's idle runner (PID 31904) was requested and denied by the permission classifier; not retried | stopping a process owned by the user's Ollama needs the user's decision | the user asks for it |
| 10 Nov 2026 | The upstream run on augmented data was stopped by the harness at step 429 of 1,700 under critically low memory; its WSL processes survived and keep running; not restarted and not killed | the harness notice says not to restart on own initiative; stopping a running job is the user's call | the user decides |
| 10 Nov 2026 | Paired comparisons use the exact McNemar test in `stair_engine.evaluation.significance`, on top-1 correctness for the same queries; discordant counts are reported | reproducible and transparent for 47 queries | a larger test set or another metric is used |
