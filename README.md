# Tocleaf

### Structure-Aware Document Intelligence

**Find the right section. Keep the evidence.**

Tocleaf is a local-first document intelligence platform built around structure-aware generative retrieval.

It implements STAIR (Structure-Aware Information Retriever), using hierarchical document structures and Table of Contents (ToC)-conditioned retrieval to identify relevant sections in large documents.

Beyond the original STAIR methodology, Tocleaf explores hybrid retrieval by combining generative section routing, BM25, dense embeddings, reranking and evidence-grounded answer generation.

## Core Capabilities

- **Hierarchical Document Understanding:** Extract, inspect and validate document structures and nested sections.
- **Structure-Aware Retrieval:** Use a fine-tuned STAIR model to identify relevant leaf sections.
- **Synthetic Dataset Generation:** Generate query-to-section training examples from document content.
- **Parameter-Efficient Fine-Tuning:** Train retrieval models using LoRA and supported local GPU configurations.
- **Hybrid Retrieval:** Combine STAIR, BM25 and dense semantic retrieval.
- **Evidence-Grounded RAG:** Generate answers with traceable document, section and page citations.
- **Retrieval Evaluation:** Compare retrieval strategies using Recall@K, MRR, nDCG and additional evaluation metrics.
- **Local-First Execution:** Keep documents, datasets, indexes and model checkpoints on your machine by default.

## Research Foundation

Tocleaf is based on:

**STAIR: A novel dataset and LLM based retriever for document structure augmentation**

Research paper: https://arxiv.org/abs/2609.03874

Official implementation: https://github.com/meghanadhpulivarthi/stair-clean

The project distinguishes the original STAIR methodology from its additional hybrid retrieval and document intelligence extensions.

## Technology Stack

- Python
- PyTorch
- Hugging Face Transformers
- PEFT / LoRA
- FastAPI
- React
- TypeScript
- SQLite
- BM25
- Dense vector retrieval

## Project Status

Under active development.

Research reproduction, implementation details, supported models and measured performance will be documented as the corresponding components are implemented and tested.

## Research Integrity

Published STAIR benchmark results are research references, not Tocleaf performance claims.

All project evaluation results will be reported from actual experiments with documented configurations and datasets.

## License

Tocleaf is released under the [MIT License](LICENSE). Copyright (c) 2026 0DevDutt0.

Third-party material is handled as follows:

- **Upstream STAIR implementation** (`meghanadhpulivarthi/stair-clean`): not included in this repository. Its authors have not published a licence, so the training path fetches it at setup (`scripts/setup_upstream.sh`), and it remains under its authors' terms. Check those terms, or ask the authors for permission, before you train models with it.
- **SearchTome benchmark**: not included. Its authors released it under CC BY-SA 4.0. Attribution and share-alike terms apply to anything derived from it.
- **Research paper**: cited, not reproduced (arXiv:2609.03874).
- **Dependencies** (PyTorch, Transformers, PEFT and others) are used under their own licences, which come with them.
