"""Command-line entry point: `uv run tocleaf <command> ...`.

Commands follow the pipeline order. Each one reads the previous step's output from disk.
Synthetic question generation reads its endpoint from STAIR_LLM_API_BASE, STAIR_LLM_API_KEY and
STAIR_LLM_MODEL. If they are not set, the command stops with a configuration error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from stair_engine.branding import PRODUCT_NAME
from stair_engine.dataset.build import SplitConfig
from stair_engine.inference.contract import load_upstream_prompt_contract
from stair_engine.structure.build import analyze_pdf
from stair_engine.training.upstream import TrainingOverrides


DEFAULT_UPSTREAM = Path.home() / "stair-work" / "upstream" / "stair-clean"


def _print_structure_summary(structure) -> None:
    print(f"{PRODUCT_NAME}: {len(structure.nodes)} ToC nodes, {len(structure.leaves)} leaves, {structure.page_count} pages")
    for issue in structure.issues:
        marker = "ERROR  " if issue.severity == "error" else "warning"
        print(f"  [{marker}] {issue.code}: {issue.message}")
    print("Validation:", "PASSED" if structure.passed else "FAILED (training blocked)")


def cmd_analyze(args) -> int:
    from stair_engine.pipeline import write_validation_report

    from stair_engine.structure.overrides import load_corrections

    corrections = load_corrections(args.corrections) if args.corrections else None
    structure = analyze_pdf(args.pdf, min_words_in_section=args.min_words, book_title=args.title, corrections=corrections)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_validation_report(structure, out / "validation.json")
    _print_structure_summary(structure)
    for node in structure.nodes:
        marker = "leaf " if node.is_leaf else "node "
        print(f"  {node.section_num:<8} {marker} p{node.page_start + 1}-{node.page_end + 1:<3} {node.word_count:>5}w  {node.title}")
    return 0 if structure.passed else 2


def cmd_build_dataset(args) -> int:
    from stair_engine.pipeline import StructureValidationError, build_dataset_from_pdf
    from stair_engine.synthetic.llm import LLMConfigurationError, client_from_environment

    try:
        # Qwen3-style thinking models spend their output budget on hidden reasoning unless told not
        # to. For the local Ollama endpoint, `reasoning_effort: "none"` is the switch that works.
        extra_body = {"reasoning_effort": args.reasoning_effort} if args.reasoning_effort else None
        client = client_from_environment(extra_body=extra_body, timeout_seconds=args.timeout)
    except LLMConfigurationError as error:
        print(f"tocleaf build-dataset: {error}", file=sys.stderr)
        return 1
    from stair_engine.structure.overrides import load_corrections

    corrections = load_corrections(args.corrections) if args.corrections else None
    try:
        manifest = build_dataset_from_pdf(
            args.pdf,
            args.workspace,
            client=client,
            questions_per_passage=args.questions_per_passage,
            max_passage_words=args.max_passage_words,
            min_words_in_section=args.min_words,
            split=SplitConfig(train=args.train, val=args.val, test=args.test, seed=args.seed),
            book_title=args.title,
            corrections=corrections,
        )
    except StructureValidationError as error:
        print(f"tocleaf build-dataset: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"dataset_id": manifest["dataset_id"], "counts": manifest["counts"], "generation": manifest["generation"]["stats"]}, indent=2))
    return 0


def _overrides_from_args(args) -> TrainingOverrides:
    return TrainingOverrides(
        model_name=args.model,
        num_train_epochs=args.epochs,
        batch_size_per_gpu=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_seq_length=args.max_seq_length,
        learning_rate=args.learning_rate,
        early_stopping_patience=args.patience,
        eval_interval=args.eval_interval,
        save_interval=args.eval_interval,
    )


def cmd_preflight(args) -> int:
    from transformers import AutoTokenizer

    from stair_engine.dataset.preflight import SequenceLengthError, check_sequence_lengths

    from stair_engine.dataset.preflight import DatasetGateError, check_dataset_gates

    overrides = _overrides_from_args(args)
    contract = load_upstream_prompt_contract(args.upstream)
    try:
        check_dataset_gates(args.data)
    except DatasetGateError as error:
        print(f"tocleaf preflight: {error}", file=sys.stderr)
        return 2
    tokenizer = AutoTokenizer.from_pretrained(overrides.model_name)
    try:
        report = check_sequence_lengths(args.data, tokenizer, contract, overrides.max_seq_length)
    except SequenceLengthError as error:
        print(f"tocleaf preflight: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2))
    return 0


def cmd_train(args) -> int:
    from stair_engine.dataset.preflight import DatasetGateError, check_dataset_gates
    from stair_engine.training.upstream import run_upstream_training

    try:
        manifest = check_dataset_gates(args.data)
    except DatasetGateError as error:
        print(f"tocleaf train: {error}", file=sys.stderr)
        return 2
    dataset_fingerprint = manifest["fingerprint_sha256"]
    contract = load_upstream_prompt_contract(args.upstream)
    result = run_upstream_training(
        upstream_root=args.upstream,
        data_dir=args.data,
        run_dir=args.run,
        overrides=_overrides_from_args(args),
        uv_binary=args.uv,
        dataset_fingerprint=dataset_fingerprint,
        prompt_contract_sha256=contract.sha256,
    )
    print(json.dumps({k: result[k] for k in ("run_id", "returncode", "duration_seconds", "peak_gpu_memory_mib", "checkpoints", "adapter_saved", "boundary_mismatch_warnings", "success")}, indent=2))
    return 0 if result["success"] else 1


def cmd_eval_stair(args) -> int:
    from stair_engine.evaluation.stair_runner import run_stair_evaluation

    contract = load_upstream_prompt_contract(args.upstream)
    result = run_stair_evaluation(
        data_dir=args.data,
        run_dir=args.run,
        contract=contract,
        base_model_name=args.model,
        out_dir=args.out,
        queries_path=args.queries,
        beams=args.beams,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        limit=args.limit,
    )
    for mode, summary in result["summaries"].items():
        print(f"[{mode}] R@1={summary.get('recall@1', 0):.3f} R@3={summary.get('recall@3', 0):.3f} "
              f"invalid={summary.get('invalid_identifier_rate', 0):.4f} non_leaf={summary.get('non_leaf_identifier_rate', 0):.4f}")
    print(json.dumps({"constraint_audit": result["constraint_audit"]}, indent=2))
    return 0


def cmd_eval_bm25(args) -> int:
    from stair_engine.pipeline import evaluate_bm25

    summary = evaluate_bm25(args.data, args.out, queries_path=args.queries)
    print(json.dumps(summary["metrics"], indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tocleaf", description=f"{PRODUCT_NAME}: STAIR retrieval pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="Validate a PDF's ToC and print the structure")
    analyze.add_argument("pdf")
    analyze.add_argument("--out", required=True)
    analyze.add_argument("--title", default=None)
    analyze.add_argument("--min-words", type=int, default=5)
    analyze.add_argument("--corrections", default=None, help="JSON file of explicit, reasoned exclusions (see corpora/)")
    analyze.set_defaults(func=cmd_analyze)

    build = sub.add_parser("build-dataset", help="Analyze, generate synthetic queries and write the dataset")
    build.add_argument("pdf")
    build.add_argument("--workspace", required=True)
    build.add_argument("--title", default=None)
    build.add_argument("--corrections", default=None, help="JSON file of explicit, reasoned exclusions")
    build.add_argument("--questions-per-passage", type=int, default=4)
    build.add_argument("--max-passage-words", type=int, default=900)
    build.add_argument("--min-words", type=int, default=5)
    build.add_argument("--train", type=float, default=0.8)
    build.add_argument("--val", type=float, default=0.1)
    build.add_argument("--test", type=float, default=0.1)
    build.add_argument("--seed", type=int, default=42)
    build.add_argument("--timeout", type=float, default=300.0)
    build.add_argument(
        "--reasoning-effort",
        default="none",
        help='Sent as reasoning_effort to the endpoint. "none" disables hidden reasoning; "" omits the field',
    )
    build.set_defaults(func=cmd_build_dataset)

    def add_training_args(p):
        p.add_argument("--upstream", default=str(DEFAULT_UPSTREAM), help="pinned upstream checkout")
        p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
        p.add_argument("--epochs", type=int, default=3)
        p.add_argument("--batch-size", type=int, default=4)
        p.add_argument("--grad-accum", type=int, default=4)
        p.add_argument("--max-seq-length", type=int, default=2048)
        p.add_argument("--learning-rate", type=float, default=1e-4)
        p.add_argument("--patience", type=int, default=5)
        p.add_argument("--eval-interval", type=int, default=30)

    preflight = sub.add_parser("preflight", help="Check every example fits max_seq_length")
    preflight.add_argument("--data", required=True)
    add_training_args(preflight)
    preflight.set_defaults(func=cmd_preflight)

    train = sub.add_parser("train", help="Run the pinned upstream trainer (needs a CUDA GPU)")
    train.add_argument("--data", required=True)
    train.add_argument("--run", required=True, help="fresh run directory")
    train.add_argument("--uv", default="uv")
    add_training_args(train)
    train.set_defaults(func=cmd_train)

    eval_stair = sub.add_parser("eval-stair", help="Evaluate a trained adapter with constrained and unconstrained decoding")
    eval_stair.add_argument("--data", required=True)
    eval_stair.add_argument("--run", required=True)
    eval_stair.add_argument("--out", required=True)
    eval_stair.add_argument("--queries", default=None, help="external query file (author-written); default: synthetic test split")
    eval_stair.add_argument("--beams", type=int, default=10)
    eval_stair.add_argument("--max-new-tokens", type=int, default=64)
    eval_stair.add_argument("--device", default="cuda")
    eval_stair.add_argument("--limit", type=int, default=None)
    eval_stair.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    eval_stair.add_argument("--upstream", default=str(DEFAULT_UPSTREAM))
    eval_stair.set_defaults(func=cmd_eval_stair)

    bm25 = sub.add_parser("eval-bm25", help="BM25 baseline over the same test queries")
    bm25.add_argument("--data", required=True)
    bm25.add_argument("--out", required=True)
    bm25.add_argument("--queries", default=None, help="external query file (author-written); default: synthetic test split")
    bm25.set_defaults(func=cmd_eval_bm25)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())


