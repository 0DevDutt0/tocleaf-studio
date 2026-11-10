# Augmented-data results: OpenStax, *Introduction to Intellectual Property*

Run on 10 Nov 2026, branch `phase-1-vertical-slice`. Every system is scored on the same 47 synthetic test queries (one per
leaf), which are the test split of the earlier run `openstax-ip-r1`.

## Summary

- **Headline:** the paper's selection rule (checkpoint chosen by constrained dev Recall@1), local trainer, augmented data:
  constrained beam R@1 **0.660** (31 of 47) and R@3 0.830. The earlier run (upstream trainer, 200 training queries) gave
  0.404 and 0.787. Exact McNemar p = 0.0018.
- **Data effect, same upstream trainer and eval-loss selection:** the upstream run on the augmented data gives R@1
  **0.574** against 0.404 on the earlier data, a gain of 17 points. 13 queries are right only in the new run and 5 only in
  the earlier one (p = 0.096). This is suggestive, not significant at 0.05.
- **Selection rule, same local run:** choosing by dev R@1 gives 0.660, against 0.426 for choosing by dev loss (p = 0.019).
  This is the one comparison in the set where a single factor changes and the difference is significant at 0.05.
- **Trainer and rule, same augmented data:** the headline against the upstream run on the augmented data is 0.660 against
  0.574 (7 against 3 queries, p = 0.34). With 47 queries this is not distinguishable.
- **BM25 remains clearly better** on these queries: R@1 0.936. The upstream run on augmented data is well below it (p < 0.001).

## Dataset

| Dataset | Train | Dev (validation) | Test |
|---|---|---|---|
| `openstax-ip` (earlier run) | 200 | 18 | 47 |
| `openstax-ip-aug` (this run) | 679 = 200 + 479 generated | 18, unchanged | 47, unchanged (byte-identical) |

Augmentation: 8 questions per passage, 527 candidates. Rejected: 39 exact duplicates of training queries and 9
near-duplicates of dev or test queries (content-token Jaccard at least 0.6). The coverage gate passed. Preflight: the
longest example is 622 tokens against a 2,048-token limit.

## Runs

| Run | Trainer | Training data | Selection | Selected checkpoint | Dev R@1 | Dev loss at selected |
|---|---|---|---|---|---|---|
| `openstax-ip-r1` (earlier) | upstream | 200 | eval loss, evaluated every 250 steps | step 250 (epoch 10) | not computed | 0.423 |
| `openstax-ip-aug-local-r1`, **headline** | local | 679 | dev constrained R@1, patience 5 (paper's rule) | epoch 14 | 0.722 | 0.200 |
| `openstax-ip-aug-local-r1`, control | local, same run | 679 | lowest dev loss | epoch 2 | 0.500 | 0.144 |
| `openstax-ip-aug-upstream-r1` | upstream | 679 | eval loss, evaluated every 85 steps | step 255 (epoch 3) | not computed | 0.127 |

The headline run trained for 19 epochs (1,615 optimizer steps) and stopped when dev R@1 had not improved for five epochs.
It took 20 minutes, with peak allocated GPU memory of 3,059 MiB. Seed 42. LoRA rank 16, alpha 32, dropout 0.1 on the
seven projection modules. AdamW with learning rate 1e-4, betas (0.9, 0.95), weight decay 0.05, 20 warmup steps. Batch
4 with gradient accumulation 2.

The upstream run on augmented data used the earlier run's schedule (20 epochs, batch 4, gradient accumulation 2, learning
rate 1e-4, patience 6, warmup 20, LoRA 16/32/0.1). Only the data changed, and the evaluation interval: 85 steps (one epoch)
instead of 250. It trained for 719 s and stopped at step 765, after six evaluations without a lower dev loss.

## Test results (47 queries)

| System | R@1 | R@3 |
|---|---|---|
| BM25 over leaves | **0.936** | **1.000** |
| Headline (paper's rule, epoch 14), constrained beam | **0.660** | 0.830 |
| Headline, constrained exact (per-step renormalised) | 0.660 | 0.830 |
| Headline, full-vocabulary exact | 0.660 | 0.830 |
| Headline, unconstrained greedy | 0.660 | 0.660 |
| Upstream trainer on augmented data (step 255), constrained beam | 0.574 | 0.787 |
| Upstream on augmented data, constrained exact | 0.553 | 0.787 |
| Upstream on augmented data, full-vocabulary exact | 0.553 | 0.787 |
| Upstream on augmented data, unconstrained greedy (training input) | 0.532 | 0.532 |
| Upstream on augmented data, unconstrained greedy (upstream prompt) | 0.553 | 0.553 |
| Control (dev-loss pick, epoch 2), constrained beam | 0.426 | 0.723 |
| Control, constrained exact | 0.404 | 0.723 |
| Earlier run `openstax-ip-r1`, constrained beam | 0.404 | 0.787 |

Every constrained output was a valid leaf: the invalid and non-leaf identifier rates are 0 in every constrained row.

## Paired tests

Exact McNemar test (two-sided) on per-query top-1 correctness with constrained beam decoding. The module is
`stair_engine.evaluation.significance`. Full output is in `significance.json`.

| Comparison | Right only in A | Right only in B | Net | p |
|---|---|---|---|---|
| Headline vs earlier run `openstax-ip-r1` | 13 | 1 | +12 | 0.0018 |
| Headline vs dev-loss control, same run | 15 | 4 | +11 | 0.019 |
| Dev-loss control vs earlier run | 9 | 8 | +1 | 1.0 |
| Upstream on augmented data vs earlier run (data only) | 13 | 5 | +8 | 0.096 |
| Headline vs upstream on augmented data (trainer and rule) | 7 | 3 | +4 | 0.34 |
| Upstream on augmented data vs BM25 | 1 | 18 | −17 | <0.001 |
| Headline vs BM25 | 1 | 14 | −13 | 0.001 |

## How to read these results

1. **Selection rule (same local run): significant.** Choosing the epoch by the paper's dev Recall@1 rule gives +23 points
   over choosing by dev loss (p = 0.019). Trainer, data and tokenisation are fixed, so only the selection differs.
2. **Data (same upstream trainer and selection rule): suggestive.** +17 points, p = 0.096. The evaluation interval changed
   from 250 to 85 steps, as planned, so the selected checkpoint moved from epoch 10 to epoch 3. Both are eval-loss picks.
3. **Trainer and rule (same augmented data): not distinguishable.** The headline is +9 points over the upstream run on the
   augmented data (p = 0.34). The local trainer's own loss-selected checkpoint (0.426) is well below the upstream run's
   loss-selected checkpoint (0.574) on the same data. Likely contributing factors are the different selected epochs and
   the training tokenisation: the upstream training input has boundary mismatches in 697 of its examples, while the local
   layout matches the decoder's tokenisation. These are candidate explanations, not tested causes.
4. **The combined gain over the earlier run is +26 points (p = 0.0018).** It is split between the data (+17, p = 0.096)
   and the trainer and rule (+9, p = 0.34). Neither part reaches significance alone at 47 queries.
5. **BM25 remains clearly better** on these synthetic queries: 0.936, against 0.660 for the headline and 0.574 for the
   upstream run on augmented data. The synthetic queries reuse their passages' wording, which favours lexical retrieval.
   An author-written query set is the fairer test of generalisation.

## Caveats

- **One seed (42).** Variance across seeds is not measured.
- **Dev set of 18 queries.** R@1 moves in steps of 5.6 points. Across epochs 10 to 19, local dev R@1 ranged from 0.556 to
  0.722, so the chosen epoch is partly chance. Its test result (0.660) is the evidence that it generalises here.
- **Shortened rule for the local trainer.** The paper uses up to 200 epochs with patience 20. This run uses up to 20 epochs
  with patience 5, and it stopped at epoch 19. A longer run could choose a different epoch.
- **Dev-loss control.** Epoch 2 is also the lowest dev loss across all 19 epochs. A loss-based rule with the upstream
  trainer's own patience would stop earlier and choose the same epoch.
- **Two trainers, not identical.** The upstream trainer's training input has boundary mismatches (697 examples on the
  augmented data, 218 on the earlier data); the local trainer's layout matches the decoder. The two runs also use different
  epoch caps and selection code. The upstream run uses the same optimiser settings as the earlier run.
- **The test queries have been used before.** Earlier configurations in this project were scored on the same 47 queries.
  The headline rule was fixed before this run's results were seen, but the test set is not held out, and the p-values do
  not correct for the earlier configurations. Treat these as exploratory until an independent query set exists.

## Upstream run: history

The run finished. Its training exited with code 0 after 719 s, and the evaluation ran automatically from the same launcher.

On 10 Nov 2026 the harness stopped the background shell that launched the run, at step 429 of 1,700, because host memory
was critically low. The WSL processes (the launcher, the trainer and its worker) survived the stop and kept running. The
run was not restarted, and no process was killed. It completed without intervention.

GPU memory: an earlier note recorded a 14.4 GB peak for the earlier upstream run, which was treated as the trainer's
requirement and called the run blocked. That figure included Ollama's idle runner (11.5 GB), which was loaded before
training began. The trainer itself needed about 2.9 GB, from the earlier run's sampled log.

Eval loss on the dev split, by step (from `upstream_trainer_state.json`): 0 → 0.835; 85 → 0.204; 170 → 0.174;
**255 → 0.127 (selected)**; 340 → 0.156; 425 → 0.168; 510 → 0.327; 595 → 0.186; 680 → 0.299; 765 → 0.287.

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
- `upstream_summary.json`: evaluation of the upstream adapter on the augmented data.
- `upstream_run_manifest.json`: upstream run configuration, return code, checkpoints and boundary-mismatch count.
- `upstream_trainer_state.json`: eval-loss history of the upstream run.
- `dataset_manifest.json`: augmented dataset manifest (counts, augmentation statistics, fingerprint).
- `significance.json`: the seven paired comparisons above.
