# Author-written query set: OpenStax, *Introduction to Intellectual Property*

Status: evaluated for training seeds 42, 43 and 44. The first seed 44 attempt stopped early and its output is not used; the seed 44 run reported here is a complete rerun with the same configuration and seed. The protocol below was fixed before any result on this set was seen.

## What this tests

Whether the earlier results hold on queries that the synthetic generator did not write: different wording, paraphrased
and contextual questions. It tests generalisation to query wording over the same 47 leaves. It does not test
generalisation to other documents.

## The query set

- `corpora/openstax-introduction-to-intellectual-property/author_queries.jsonl`, SHA-256
  `060c93c45d7137d3deeb30acd93c61fbb61c627ab2ed9f29e4bd58b3e09b0ea8`, frozen before any run.
- 141 answerable queries (47 leaves, each with a `direct`, a `paraphrase` and a `contextual` query) and 6 out-of-scope
  queries. The out-of-scope queries are not in the ranking metrics.
- Written by the project's engineering agent, not an independent annotator. The writer read the opening text of each
  leaf (about 260 words) and the ToC. It did not read the synthetic queries while writing, and it had seen no results
  on this set.
- Leakage screen after writing: seven queries had a content-token Jaccard of at least 0.6 with a synthetic question
  (train, validation or test; 744 in all). They were rewritten. The originals, replacements and matched synthetic
  questions are in `screening_log.json`. After the rewrite, no query reaches 0.6. The screen summary is in
  `screen_summary.json`.
- Lexical overlap with the gold leaf. Mean share of query content words found in the leaf: 0.692 for this set (direct
  0.842, paraphrase 0.670, contextual 0.564), against 0.839 for the synthetic test split. Share of queries whose gold leaf
  has the highest overlap among the 47 leaves: 0.645 for this set, 0.872 for the synthetic test split. The set is less
  lexically aligned with its passages than the synthetic split, as intended. Overlap does not measure retrieval
  difficulty, and it was not used to choose or edit any query beyond the leakage rewrite.

## Protocol (fixed before results)

1. **Primary metric:** constrained beam Recall@1 over the 141 answerable queries, for the headline configuration (local
   trainer, the paper's selection rule of dev Recall@1, augmented data, seed 42). The comparison is BM25 over leaves on
   the same queries, with an exact McNemar test on per-query top-1 correctness.
2. **Seeds:** the same configuration with seeds 43 and 44, each a fresh run. Report each seed's value, the mean and the
   range. Three seeds do not support a valid confidence interval, so no p-value is reported across seeds.
3. **Context:** constrained exact and full-vocabulary exact, Recall@3, the dev-loss pick of each run, the upstream
   trainer on augmented data, and the earlier run (200 training queries). The synthetic test split is context only: it
   has been used in earlier configurations.
4. No query, configuration or analysis is changed after this point. A change needs a new set name and a new freeze.

## Commands

From the project root in WSL, with the project environment (`UV_PROJECT_ENVIRONMENT=~/.venvs/tocleaf`). The paths are the ones used for these runs, under `~/stair-work` in WSL.

```
tocleaf eval-bm25 --data ~/stair-work/workspace/openstax-ip-aug/dataset --out ~/stair-work/evals/openstax-ip-author-bm25 --queries corpora/openstax-introduction-to-intellectual-property/author_queries.jsonl
tocleaf eval-stair --data ~/stair-work/workspace/openstax-ip-aug/dataset --run ~/stair-work/runs/openstax-ip-aug-local-r1 --out ~/stair-work/evals/openstax-ip-author-local-r1 --queries corpora/openstax-introduction-to-intellectual-property/author_queries.jsonl --device cuda
tocleaf train-local --data ~/stair-work/workspace/openstax-ip-aug/dataset --run ~/stair-work/runs/openstax-ip-aug-local-s43 --device cuda --seed 43
tocleaf train-local --data ~/stair-work/workspace/openstax-ip-aug/dataset --run ~/stair-work/runs/openstax-ip-aug-local-s44 --device cuda --seed 44
python -m stair_engine.evaluation.significance ~/stair-work/evals/openstax-ip-author-bm25/bm25_per_query.jsonl ~/stair-work/evals/openstax-ip-author-local-r1/per_query.jsonl --mode constrained
```

## Results

The table uses training seed 42 for every STAIR system, and one BM25 run. Seeds 43 and 44 are in the training-seeds table below. All scores are on the 141 answerable queries. Recall@1
is the constrained-beam top-1 unless stated. Per-query files stay out of the repository. Sanitised summaries are in
`summaries/`, and the numbers are in `results_table.json` and `significance.json`.

### Systems on the author set

| System | R@1 | R@3 | direct R@1 | paraphrase R@1 | contextual R@1 |
|---|---|---|---|---|---|
| BM25 over leaves | **0.780** | **0.936** | 0.894 | 0.723 | 0.723 |
| STAIR headline: dev R@1 rule pick (seed 42) | 0.667 | 0.844 | 0.872 | 0.574 | 0.553 |
| STAIR dev-loss control (same run) | 0.447 | 0.773 | 0.553 | 0.319 | 0.468 |
| STAIR upstream trainer, augmented data (eval-loss pick) | 0.482 | 0.766 | 0.638 | 0.404 | 0.404 |
| STAIR earlier run, 200 training queries | 0.468 | 0.674 | 0.660 | 0.340 | 0.404 |

Other decodings of the headline: constrained exact R@1 0.681 (R@3 0.837); full-vocabulary exact R@1 0.667; unconstrained
greedy R@1 0.667. For all four STAIR systems, every constrained output was a valid leaf: 0 invalid and 0 non-leaf
identifiers over 6,627 beams each.

### Training seeds (rule pick, constrained beam)

| Seed | Author set R@1 | Author set R@3 | Synthetic test R@1 | Per-seed paired p vs BM25 |
|---|---|---|---|---|
| 42 | 0.667 | 0.844 | 0.660 | 0.023 |
| 43 | 0.681 | 0.809 | 0.617 | 0.076 |
| 44 | 0.624 | 0.787 | 0.681 | 0.0016 |
| Mean (range) | 0.657 (0.624 to 0.681) | 0.813 (0.787 to 0.844) | 0.652 (0.617 to 0.681) | not reported across seeds |

Seed 42 selected epoch 14, seed 43 epoch 11 and seed 44 epoch 14. Their dev Recall@1 at the selected epoch was 0.722, 0.722 and 0.778 (18 dev queries, so one query is 5.6 points). For seeds 43 and 44 the rule and the dev-loss pick are the same epoch, so their dev-loss rows are the same checkpoint and score the same as the rule rows (0.681 and 0.624). For seed 44 the two checkpoints are byte-identical. The three seeds differ by 5.7 points on the author set. BM25 scores 0.780 on the same queries. The per-seed p-values describe each seed separately: the gap to BM25 is significant at 0.05 for seeds 42 and 44 and not for seed 43. No p-value is reported across seeds.

### Paired exact McNemar tests (constrained-beam top-1)

| Comparison | R@1, A vs B | A right, B wrong | B right, A wrong | Net | Two-sided p |
|---|---|---|---|---|---|
| BM25 vs headline (primary) | 0.780 vs 0.667 | 30 | 14 | +16 | 0.023 |
| Headline vs dev-loss control | 0.667 vs 0.447 | 38 | 7 | +31 | 3e-6 |
| Headline vs upstream trainer, augmented | 0.667 vs 0.482 | 30 | 4 | +26 | 6e-6 |
| Headline vs earlier run | 0.667 vs 0.468 | 39 | 11 | +28 | 9e-5 |
| Upstream augmented vs earlier run (data only) | 0.482 vs 0.468 | 28 | 26 | +2 | 0.89 |
| BM25 vs upstream augmented | 0.780 vs 0.482 | 53 | 11 | +42 | 1e-7 |
| BM25 vs earlier run | 0.780 vs 0.468 | 58 | 14 | +44 | 2e-7 |
| BM25 vs dev-loss control | 0.780 vs 0.447 | 56 | 9 | +47 | 2e-9 |

### The same adapters on both query sets (constrained-beam R@1)

| System | Synthetic test, 47 queries | Author set, 141 queries | Change, points |
|---|---|---|---|
| BM25 | 0.936 | 0.780 | -15.6 |
| STAIR headline | 0.660 | 0.667 | +0.7 |
| STAIR dev-loss control | 0.426 | 0.447 | +2.1 |
| STAIR upstream, augmented | 0.574 | 0.482 | -9.2 |
| STAIR earlier run | 0.404 | 0.468 | +6.4 |

The synthetic figures come from `docs/results/openstax-ip-aug/README.md`.

### What the results say

1. **Primary comparison.** BM25 is ahead of the headline on this set, 0.780 against 0.667 (p = 0.023 for seed 42, the pre-specified headline). Seeds 43 and 44 point the same way: BM25 leads by 10 points (p = 0.076, not significant) and 16 points (p = 0.0016). On the
   synthetic split the seed 42 lead was about 28 points, and here it is about 11. The gap is smaller mainly because BM25 drops by
   16 points on this set, while the headline does not change.
2. **Selection rule.** The headline beats the dev-loss pick of the same run by 22 points on this set (p = 3e-6). On the
   synthetic split the margin was 23 points, so the rule's advantage holds on both query sets. On seeds 43 and 44 the two rules picked the same epoch, so that comparison does not exist for them.
3. **Augmentation.** On this set the augmented data gives no measurable gain. The upstream trainer on augmented data
   scores 0.482 and the same trainer on the earlier 200 queries scores 0.468 (p = 0.89). On the synthetic split the same
   comparison gave +17 points (p = 0.096). The gain does not carry over to independently worded queries. That suggests it
   reflects the generator's query style, but this set cannot test that directly.
4. **Trainer and rule together.** The headline beats the upstream run on augmented data by 18 points (p = 6e-6). That
   comparison changes both the trainer and the selection rule, so it cannot separate them. The upstream run's eval-loss
   pick (0.482) is close to the local dev-loss pick (0.447), which points to the selection rule rather than the trainer.
   That is an inference from two runs.
5. **Where BM25 leads.** On direct queries the headline is close to BM25 (0.872 against 0.894). The gap is in paraphrase
   (0.574 against 0.723) and contextual queries (0.553 against 0.723).

### Limits

- The author set is agent-written, not an independent annotation. The writer read only the first ~260 words of each leaf.
  Seven queries were rewritten after the leakage screen. That screen compared text only; no model or BM25 result was used.
- Three training seeds (42, 43 and 44) are complete. The first seed 44 attempt stopped soon after writing its first-epoch checkpoints; its cause is not confirmed, and its output is not used. The seed 44 run is a full rerun with the same configuration and seed,
  so seed variance rests on three runs, which differ by 5.7 points. Dev Recall@1 for seed 44 moved between 0.500 and 0.778 over epochs 2 to 19. The dev set has 18 queries, so the selected checkpoint is noisy.
- The test covers new query wording over the same 47 leaves. It does not cover a new document.
- Eight paired tests are reported. Only the first, BM25 against the headline, was pre-specified as the primary test. The
  others are exploratory and not corrected for multiple comparisons. The per-seed p-values for seeds 43 and 44 follow protocol item 2 and are not corrected either.

## Files

- `results_table.json`: the tables above, machine-readable.
- `significance.json`: the eight paired tests, with unrounded exact two-sided p-values.
- `summaries/`: sanitised evaluation summaries for BM25, the headline, the dev-loss control, the upstream run on augmented
  data and the earlier run. For seeds 43 and 44 it also holds the rule pick, the dev-loss pick and the synthetic test.
- `screening_log.json`: the seven rewrites, with the original and replacement text and the matched synthetic question.
- `screen_summary.json`: the post-rewrite screen (leakage flags, overlap statistics by query type).
- `seed43_results.json` and `seed44_results.json`: each seed's results on both query sets, with its per-seed paired test against BM25.
