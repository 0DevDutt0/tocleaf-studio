# Tocleaf

**Find the right section. Keep the evidence.**

Tocleaf is a local research prototype. Give it a PDF with a bookmark outline (a table of contents), and it learns to name the section that answers a question. It implements the STAIR method (arXiv:2609.03874) end to end: structure extraction, synthetic training data, LoRA fine-tuning, constrained decoding and evaluation against a keyword baseline.

## Results

The main test is a 201-page open textbook with 47 sections: OpenStax, *Introduction to Intellectual Property* (CC BY 4.0). The test has 141 questions. An AI coding agent wrote them separately from the training data generator, and they were frozen before any model was scored. Recall@1 is the share of questions whose correct section is ranked first.

| System | Recall@1 | Recall@3 |
|---|---|---|
| BM25 keyword baseline | 0.780 | 0.936 |
| STAIR, checkpoint chosen by dev Recall@1 | 0.667 | 0.844 |
| STAIR, checkpoint chosen by dev loss | 0.447 | 0.773 |

On this test the keyword baseline is still ahead on all three seeds. On seed 42 the gap is 11 points (paired exact test p = 0.023). On seed 43 it is 10 points (p = 0.076, not significant). On seed 44 it is 16 points (p = 0.0016). An earlier test built from generated questions showed a bigger gap (0.936 against 0.660). Those questions reuse the wording of their passages, which suits keyword matching, so a separate test was written to check it.

What the runs show:

- Constrained decoding does what it should. In every STAIR evaluation, every output was a valid leaf section: 0 invalid and 0 non-leaf identifiers across all constrained beams.
- The checkpoint rule matters, on seed 42. Choosing by dev Recall@1 beats choosing by dev loss by 22 points (p < 0.001). On seeds 43 and 44 both rules picked the same epoch, so that comparison does not exist for them.
- Augmented training data gave a gain on the generated test (+17 points, p = 0.096, not significant at 0.05). It gave no gain on the separate test (0.482 against 0.468, p = 0.89).
- Three training seeds give 0.667 (seed 42), 0.681 (seed 43) and 0.624 (seed 44) on the separate test, a mean of 0.657. The synthetic test gives 0.660, 0.617 and 0.681. The first attempt at seed 44 stopped early and was rerun in full. Seed details are in `docs/results/openstax-ip-author/`.

## How it works

1. **Structure.** Reads the PDF's outline, checks numbering and nesting, and removes running headers that repeat on each page. Validation can stop the run. Each manual correction needs a written reason.
2. **Training data.** A local model (qwen3.5:9b through Ollama) writes questions for each section. Generated questions that closely copy a held-out question are dropped.
3. **Model.** LoRA fine-tuning of Qwen2.5-0.5B-Instruct. In constrained mode, the decoder can only emit identifiers from the section tree.
4. **Evaluation.** Recall@K, MRR and nDCG, a BM25 baseline over the same sections, paired exact tests, and an audit of every constrained beam.
5. **Records.** Dataset and run manifests keep the configuration, a SHA-256 fingerprint of the dataset, and peak GPU memory. Frozen query sets are identified by hash.

## Stack and hardware

- Python 3.12, PyTorch 2.14, Hugging Face Transformers 5.17, PEFT for LoRA, pypdfium2 for PDF text, and an OpenAI-compatible client for the local model server.
- Developed on Windows 11 with WSL2 (Ubuntu) and an NVIDIA RTX 5090 Laptop GPU with 24 GB of memory. The main training run took about 20 minutes and peaked at about 3 GB of GPU memory.

## Quick start

Run the setup in WSL or Linux. Training needs an NVIDIA GPU.

```bash
scripts/setup_env.sh          # project environment
scripts/setup_upstream.sh     # pinned upstream checkout used by the training path

uv run tocleaf analyze path/to/book.pdf --out workspace/mybook-analysis

export STAIR_LLM_API_BASE=http://127.0.0.1:11434/v1   # local Ollama server
export STAIR_LLM_MODEL=qwen3.5:9b
uv run tocleaf build-dataset path/to/book.pdf --workspace workspace/mybook --title "My Book"
uv run tocleaf preflight --data workspace/mybook/dataset
uv run tocleaf train-local --data workspace/mybook/dataset --run runs/mybook --device cuda
uv run tocleaf eval-stair --data workspace/mybook/dataset --run runs/mybook --out evals/mybook --device cuda
uv run tocleaf eval-bm25 --data workspace/mybook/dataset --out evals/mybook-bm25
```

`uv run pytest` runs 81 unit tests. All 81 passed in about six seconds on CPU, with no GPU or Ollama needed.

## Repository layout

- `src/stair_engine/`: structure extraction, dataset builder, constrained decoding, trainers and evaluation
- `tests/`: unit tests for structure, corrections, dataset gates, constrained decoding, trainers and the significance test
- `corpora/`: provenance and correction files for the test documents. The PDFs are not committed.
- `docs/results/`: evaluation summaries, manifests and the statistics behind the numbers above
- `docs/`: research notes, architecture decisions and the implementation plan
- `scripts/`: environment setup

## Status

Working end to end on the textbook above: structure validation, dataset gates, training and evaluation. The STAIR paper itself ran through the same chain as a smoke test (11 pages, 51 queries). The model had too little training data there to score well.

Not built yet: dense embeddings, hybrid retrieval, reranking, answers with citations, an API and a web interface. `IMPLEMENTATION_PLAN.md` and `docs/architecture/` describe them as planned work. They are not features of this code.

## Limitations

- The main test is one textbook. Its queries were written by an AI coding agent, not an independent annotator, and the writer read only the opening text of each section.
- Results come from three training seeds. The first attempt at seed 44 stopped early and was rerun in full; its cause is not confirmed.
- The model is small (0.5B parameters). Keyword matching is hard to beat when a question reuses the section's own wording. The paraphrased and contextual questions show where the model still loses.
- The training path runs an upstream research repository that has no licence. See the licence section.
- A PDF without a native outline is refused. The pipeline does not infer structure from headings.

## Research basis

STAIR: A novel dataset and LLM based retriever for document structure augmentation. arXiv:2609.03874 (CC BY 4.0). Official implementation: https://github.com/meghanadhpulivarthi/stair-clean

## Research integrity

Published STAIR benchmark numbers are references, not claims about this code. The numbers in this README come from result files in `docs/results/`, and each file records its configuration and dataset.

## License

Tocleaf is released under the [MIT License](LICENSE). Copyright (c) 2026 0DevDutt0.

Third-party material is handled as follows:

- **Upstream STAIR implementation** (`meghanadhpulivarthi/stair-clean`): not included in this repository. Its authors have not published a licence, so the training path fetches it at setup (`scripts/setup_upstream.sh`), and it remains under its authors' terms. Check those terms, or ask the authors for permission, before you train models with it.
- **SearchTome benchmark**: not included. Its authors released it under CC BY-SA 4.0. Attribution and share-alike terms apply to anything derived from it.
- **Research paper**: cited, not reproduced (arXiv:2609.03874).
- **Dependencies** (PyTorch, Transformers, PEFT and others) are used under their own licences, which come with them.
