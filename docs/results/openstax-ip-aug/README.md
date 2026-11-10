# Augmented-data results: OpenStax, *Introduction to Intellectual Property*

Run on 10 Nov 2026, branch `phase-1-vertical-slice`. Every system is scored on the same 47 synthetic test queries (one per
leaf), which are the test split of the earlier run `openstax-ip-r1`.

## Summary

- **Headline:** the paper's selection rule (checkpoint chosen by constrained dev Recall@1) on the augmented data gives
  constrained beam R@1 **0.660** (31 of 47) and R@3 0.830. The earlier run, on 200 training queries, gave 0.404 and 0.787.
  Exact McNemar p = 0.0018 (13 queries right only in the headline run, 1 only in the earlier run).
- **Selection rule, same training run:** choosing the epoch by dev Recall@1 gives 0.660, against 0.426 for choosing by
  dev loss. p = 0.019 (15 against 4). Trainer, data and tokenisation are identical in these two rows.
- **Data effect:** not yet isolated. The earlier run used a different trainer, tokenisation and schedule. The upstream
  trainer on the augmented data, with the earlier run's schedule, is the comparison that isolates it.
- **Upstream run on augmented data: not finished.** The harness stopped its background shell at step 429 of 1,700 when
  host memory was critically low. Its WSL processes kept running, and whether to stop them or let them finish is pending.
- **BM25 is still stronger** on these queries: R@1 0.936. The headline is 27.6 points below it (p = 0.001).

## Dataset

| Dataset | Train | Dev (validation) | Test |
|---|---|---|---|
| `openstax-ip` (earlier run) | 200 | 18 | 47 |
| `openstax-ip-aug` (this run) | 679 = 200 + 479 generated | 18, unchanged | 47, unchanged (byte-identical) |

Augmentation: 8 questions per passage, 527 candidates. Rejected: 39 exact duplicates of training queries and 9
near-duplicates of dev or test queries (content-token Jaccard at least 0.6). The coverage gate passed. Preflight: the
longest example is 622 tokens against a 2,048-token limit.

## Runs

| Run | Trainer | Training data | Selection | Epoch | Dev R@1 | Dev loss |
|---|---|---|---|---|---|---|
| `openstax-ip-r1` (earlier) | upstream | 200 | eval loss, evaluated every 250 steps | 10 (step 250) | not computed | 0.423 |
| `openstax-ip-aug-local-r1`, **headline** | local | 679 | dev constrained R@1, patience 5 (paper's rule) | 14 | 0.722 | 0.200 |
| `openstax-ip-aug-local-r1`, control | local, same run | 679 | lowest dev loss | 2 | 0.500 | 0.144 |
| `openstax-ip-aug-upstream-r1` | upstream | 679 | eval loss, earlier schedule, evaluated every 85 steps | not finished (step 429 of 1,700) | — | — |

The headline run trained for 19 epochs (1,615 optimizer steps) and stopped when dev R@1 had not improved for five epochs.
It took 20 minutes, with peak allocated GPU memory of 3,059 MiB. Seed 42. LoRA rank 16, alpha 32, dropout 0.1 on the
seven projection modules. AdamW with learning rate 1e-4, betas (0.9, 0.95), weight decay 0.05, 20 warmup steps. Batch
4 with gradient accumulation 2.

## Test results (47 queries)

| System | R@1 | R@3 |
|---|---|---|
| BM25 over leaves | **0.936** | **1.000** |
| Headline (paper's rule, epoch 14), constrained beam | **0.660** | 0.830 |
| Headline, constrained exact (per-step renormalised) | 0.660 | 0.830 |
| Headline, full-vocabulary exact | 0.660 | 0.830 |
| Headline, unconstrained greedy | 0.660 | 0.660 |
| Control (dev-loss pick, epoch 2), constrained beam | 0.426 | 0.723 |
| Control, constrained exact | 0.404 | 0.723 |
| Earlier run `openstax-ip-r1`, constrained beam | 0.404 | 0.787 |
| Upstream trainer on augmented data | not finished | not finished |

Every constrained output was a valid leaf: the invalid and non-leaf identifier rates are 0 in every constrained row.

## Paired tests

Exact McNemar test (two-sided) on per-query top-1 correctness with constrained beam decoding. The module is
`stair_engine.evaluation.significance`. Full output is in `significance.json`.

| Comparison | Right only in A | Right only in B | Net | p |
|---|---|---|---|---|
| Headline vs earlier run `openstax-ip-r1` | 13 | 1 | +12 | 0.0018 |
| Headline vs dev-loss control, same run | 15 | 4 | +11 | 0.019 |
| Dev-loss control vs earlier run | 9 | 8 | +1 | 1.0 |
| Headline vs BM25 | 1 | 14 | −13 | 0.001 |

## How to read these results

1. **Selection rule (within one run).** Choosing the epoch by the paper's dev Recall@1 rule gives +23 points over choosing
   by dev loss (p = 0.019). Only the selection differs between these two rows, so this is the cleanest evidence in the
   set. It rests on one seed and 47 test queries.
2. **Headline against the earlier run.** The headline is +26 points (p = 0.002). That comparison changes four things at
   once: the data (479 generated queries), the trainer (local instead of upstream), the training tokenisation (the earlier
   run logged 218 boundary-mismatch warnings; the local trainer matches the decoder's tokenisation), and the selection rule.
   It does not attribute the gain to the extra queries.
3. **Dev-loss control against the earlier run.** The control scores 0.426 and the earlier run 0.404 (p = 1.0). Across
   these two rows, the augmented data has not shown a detectable effect under loss selection. The trainers differ, so
   this is a weak inference. The upstream run on augmented data is the proper test, and it is not finished.
4. **BM25.** STAIR remains far below BM25 on this benchmark. The synthetic test queries reuse their passages' wording,
   which favours lexical retrieval. An author-written query set is a fairer test of generalisation.

## Caveats

- **One seed (42).** Variance across seeds is not measured.
- **Dev set of 18 queries.** R@1 moves in steps of 5.6 points. Across epochs 10 to 19, dev R@1 ranged from 0.556 to
  0.722, so the chosen epoch is partly chance. Its test result (0.660) is the evidence that it generalises here.
- **Shortened rule.** The paper uses up to 200 epochs with patience 20. This run uses up to 20 epochs with patience 5, and
  it stopped at epoch 19. A longer run could choose a different epoch.
- **Dev-loss control.** Epoch 2 is also the lowest dev loss across all 19 epochs. A loss-based rule with the upstream
  trainer's own patience would stop earlier and choose the same epoch.
- **The test queries have been used before.** Earlier configurations in this project were scored on the same 47 queries.
  The headline rule was fixed before this run's results were seen, but the test set is not held out, and the p-values do
  not correct for the earlier configurations. Treat these as exploratory until an independent query set exists.

## Upstream trainer on the augmented data

This run isolates the data effect. It uses the upstream trainer with the earlier run's schedule (20 epochs, batch 4,
gradient accumulation 2, learning rate 1e-4, patience 6, warmup 20, LoRA 16/32/0.1). The only changes are the data and
the evaluation interval, which is one epoch (85 optimizer steps) instead of 250 steps.

**Status (10 Nov 2026): not finished.** An earlier plan said this run was blocked, because the earlier run's peak GPU
use was 14.4 GB. That figure included Ollama's idle runner, which held 11.5 GB before training began, so the trainer
itself needs about 2.9 GB. The run then started. At step 429 of 1,700 the harness stopped its background shell under
critically low host memory. The WSL processes, the trainer and the launcher, kept running. The evaluation step is part of
that launcher, so it starts automatically when training ends. Whether to stop the processes or let them finish is pending.

## Reproduce

Commands from the project root in WSL, using the project environment at `~/.venvs/tocleaf`. Paths are placeholders.

```
uv run tocleaf augment-dataset <pdf> --base <openstax-ip dataset> --out <openstax-ip-aug dataset> \
    --title "Introduction to Intellectual Property" --corrections corpora/openstax-introduction-to-intellectual-property/corrections.json \
    --questions-per-passage 8 --max-passage-words 1500
uv run tocleaf preflight --data <openstax-ip-aug dataset>
uv run tocleaf train-local --data <openstax-ip-aug dataset> --run <run dir> --device cuda
uv run tocleaf eval-stair --data <openstax-ip-aug dataset> --run <run dir> --out <eval dir> --device cuda
# Control: the dev-loss adapter in a sibling run directory, so eval-stair loads it as checkpoint-best.
mkdir <run dir>-by-loss && cp -r <run dir>/checkpoint-best-loss <run dir>-by-loss/checkpoint-best
uv run tocleaf eval-stair --data <openstax-ip-aug dataset> --run <run dir>-by-loss --out <eval dir>-by-loss --device cuda
# Upstream trainer on the augmented data
uv run tocleaf train --data <openstax-ip-aug dataset> --run <upstream run dir> --uv ~/.local/bin/uv \
    --epochs 20 --batch-size 4 --grad-accum 2 --max-seq-length 2048 --learning-rate 1e-4 --patience 6 --eval-interval 85
uv run tocleaf eval-stair --data <openstax-ip-aug dataset> --run <upstream run dir> --out <eval dir>-upstream --device cuda
# Paired comparisons
uv run python -m stair_engine.evaluation.significance <A>/per_query.jsonl <B>/per_query.jsonl --mode constrained
```

Per-query files (`per_query.jsonl`) and checkpoints are kept out of the repository, as for the earlier runs. The copied
files here have home-directory paths replaced by `<wsl-home>` and `<user>`.

## Files

- `local_rule_pick_summary.json`: evaluation of the headline adapter (summaries, constraint audit, per-type results).
- `local_loss_control_summary.json`: evaluation of the dev-loss control adapter.
- `local_run_manifest.json`: training configuration, per-epoch history, and both selections.
- `dataset_manifest.json`: augmented dataset manifest (counts, augmentation statistics, fingerprint).
- `significance.json`: the paired comparisons above.
