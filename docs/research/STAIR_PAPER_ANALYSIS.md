# STAIR paper analysis

Source: arXiv:2609.03874 (v2, dated 13 Oct 2026), read from the HTML version at
`https://arxiv.org/html/2609.03874`. Quotations below come from that text.

This document records what the paper specifies, what it leaves out, and where its own text is
inconsistent. Anything not listed under "specified" is an engineering choice made by this project
and must be labelled as such in results.

## 1. Problem

Given a document D, its table of contents ToC(D) and a query q, retrieve the correct **leaf** node
T_l. Formally `(D, ToC(D), q) -> T_l`. The model returns a section title, and the title identifies
a section whose content answers the query.

## 2. Leaf definition: an inconsistency in the paper

Section 4.1 defines the leaf set as `LN_D = { T_l in ToC_D | exists! T_c in ToC_D, T_l = parent_of(T_c) }`.
As written, this selects nodes that *have* exactly one child, which is the opposite of a leaf.

The prose ("a leaf node is a node that does not have child sections") and the examples (Table 3,
where "leaf" labels fall on nodes such as "2.2.3 Initiative vs. Guilt" and not on "2.2 Psychosocial
Theory") both describe leaves as nodes with **no children**. This project uses that definition.
Structure code computes `is_leaf` from the tree, so content length cannot change leaf status.

## 3. Specified in the paper

| Item | Paper statement |
|---|---|
| Base model | Mistral-7B-Instruct-v0.2 (Section 5.2) |
| Adaptation | LoRA, r = 16, alpha = 32 |
| Epochs and early stopping | up to 200 epochs; early stop with patience 20 epochs, using **Recall@1 on the dev set** |
| Input length | max 14k tokens for STAIR (512 for DSI) |
| Output length | max 64 tokens |
| Training input | prompt, complete ToC, query (Section 4.2; Table 2) |
| Training target | the leaf section name (Table 2: `12.9 Step seven: design course structure and learning`) |
| Inference | "constrained generation, restricting the output vocabulary to valid ToC leaf nodes" (Figure 2 caption, Stage 3) |
| Question generation | Mixtral 8x7B asked, per paragraph, for multiple questions "covering all important topics" (Section 3.2) |
| Corpus parsing | PyMuPDF to extract the ToC (Section 3.2) |
| Dev/test protocol | a portion of generated questions for dev, a small portion for training, and "majority" for test (Section 3.2); per-book counts in Table 1 |
| Metrics | Recall@1, Recall@3, nDCG@3 via BEIR code (Section 5); randomisation test for significance (Section 6) |
| Hallucination definition | "the generation of a non-leaf node (invalid document identifier)" (Section 6.1) |

## 4. Not specified in the paper

These are missing and must be chosen and recorded by this project:

- **Constrained decoding mechanism.** Section 4.2 states the restriction but gives no algorithm (no
  trie, no logits mask, no beam settings). Section 5.1 mentions beam search only for the Mistral
  baseline. This project implements a prefix trie over the exact training completion tokens, with
  beam search (see ADR-006).
- Learning rate, batch size, optimiser, dropout, target modules, precision, seeds, gradient
  accumulation, warm-up.
- Number of questions per paragraph, the question-generation prompt, and how duplicate or unanswerable
  questions were handled.
- How the leaf sections were built from PDF bookmarks (only "parsed the PDF ... extracted Table of
  Contents using pymupdf").
- How multiple relevant leaves are scored (the paper's examples use one gold leaf).
- The exact number of dev and test questions per book beyond Table 1.

## 5. Internal inconsistencies to note

1. **Leaf formula** (section 2 above).
2. **Haiku baseline.** Table 4's caption includes "Haiku is Claude haiku 4.5", and a Haiku row exists
   with averages R@1 45.6, R@3 64.1, nDCG@3 56.4. Section 5.1 does not describe that baseline.
3. **Prompt format.** Table 2 shows the prompt "Pick the best section ... Only generate the section
   name" with `leaf` markers in the ToC. The upstream implementation uses a different prompt and a
   list-shaped target (see the upstream audit). Those differences are recorded there.
4. **Mistral error breakdown.** Section 6.2 says Mistral's error rate is 86.20%, of which 26.81%
   are non-leaf and 23.86% are hallucinations. The two categories are not defined consistently in the
   text, so these figures cannot be reproduced without the paper's evaluation code.

## 6. Reported results (Table 4, average over six domains)

| System | R@1 | R@3 | nDCG@3 |
|---|---|---|---|
| Mistral (out of the box, beam search) | 13.8 | 16.4 | 15.4 |
| BM25 | 59.5 | 77.6 | 70.1 |
| DPR (NV-Embed-v2) | 68.7 | 85.4 | 78.6 |
| DSI | 76.9 | 85.3 | 81.9 |
| **STAIR** | **82.6** | **90.8** | **87.5** |

Reported non-leaf rate for STAIR: 0.05% of predictions (Section 6.2). The paper does not report a
final-answer hallucination rate. The 0.05% figure must never be presented as one.

These numbers are the paper's claims on the paper's benchmark. This project has not reproduced them
(see `docs/research/SOURCING_AND_DATA_AUDIT.md`: the public SearchTome release does not match the
paper's benchmark), and it must not present them as its own results.

## 7. Limitations the paper states

- Evaluation only covers corpora that already have a global structure (Section 8).
- Future work includes dynamic ToC construction and multi-hop retrieval.

## 8. Consequences for this project

- Leaves are defined by the tree, not by content (section 2).
- Constrained decoding must be built here, because the upstream code does not contain it.
- Recall@1 early stopping and the 200-epoch budget are paper settings. Upstream uses eval loss and a
  shorter run. Any run that departs from the paper is labelled as a deviation.
- The "non-leaf rate" is reported under that name. "Hallucination" is used only where quoting the paper.
