"""Run a trained STAIR adapter over a query set with three decoding modes.

* ``constrained``: beam search restricted to valid leaf completions (paper, Stage 3). Beam width equals
  the number of returned candidates, so the ranking is the beam ranking.
* ``unconstrained``: greedy decoding with no restriction, from the input the model was trained on.
* ``unconstrained_upstream_prompt``: greedy decoding from the prompt as upstream `stair eval` tokenises
  it. It differs from training by the boundary token described in contract.py, and it is reported
  separately so that effect can be measured.

Every query is classified, and the constraint audit is recorded. A constrained run is only valid when
its audit shows zero violations.

The trained input length depends on each question, so the prefix is computed per query. Only the target
tokens are shared by every query. That is checked on real queries before any scoring.
"""

from __future__ import annotations

import datetime as dt
import json
import random
import time
from pathlib import Path

from stair_engine.evaluation.metrics import classify_output, summarise_by_type, summarise_queries
from stair_engine.evaluation.queries import load_eval_records
from stair_engine.inference.constrained import (
    ConstraintAudit,
    TokenTrie,
    make_prefix_allowed_tokens_fn,
    sequences_to_leaves,
)
from stair_engine.inference.contract import (
    PromptContract,
    completion_text,
    load_toc_view,
    render_messages,
    training_token_layout,
)

MODES = ("constrained", "unconstrained", "unconstrained_upstream_prompt")


def build_leaf_trie(tokenizer, view, probe_prompt: str) -> tuple[TokenTrie, dict[str, list[int]]]:
    """Trie over the target tokens the trainer used. Those tokens do not depend on the question."""

    sequences: dict[str, list[int]] = {}
    shared_prefix: list[int] | None = None
    for leaf_id in sorted(view.leaf_ids):
        layout = training_token_layout(tokenizer, probe_prompt, completion_text([view.label_by_id[leaf_id]]))
        if shared_prefix is None:
            shared_prefix = layout.prefix_ids
        elif layout.prefix_ids != shared_prefix:
            raise ValueError("leaf completions do not share one training prefix; constrained decoding is undefined")
        sequences[leaf_id] = layout.target_ids
    if shared_prefix is None:
        raise ValueError("the ToC has no leaves to decode")
    return TokenTrie(sequences), sequences


def _summaries(rows: list[dict]) -> dict[str, dict]:
    return {
        "constrained": summarise_queries(
            [{"ranked": r["modes"]["constrained"]["ranked"], "gold": r["gold"], "top_status": r["modes"]["constrained"]["top_status"], "items": r["modes"]["constrained"]["items"]} for r in rows]
        ),
        "unconstrained": summarise_queries(
            [{"ranked": r["modes"]["unconstrained"]["ranked"], "gold": r["gold"], "top_status": r["modes"]["unconstrained"]["top_status"], "items": r["modes"]["unconstrained"]["items"]} for r in rows]
        ),
        "unconstrained_upstream_prompt": summarise_queries(
            [{"ranked": r["modes"]["unconstrained_upstream_prompt"]["ranked"], "gold": r["gold"], "top_status": r["modes"]["unconstrained_upstream_prompt"]["top_status"], "items": r["modes"]["unconstrained_upstream_prompt"]["items"]} for r in rows]
        ),
    }


def run_stair_evaluation(
    *,
    data_dir: str | Path,
    run_dir: str | Path,
    contract: PromptContract,
    base_model_name: str,
    out_dir: str | Path,
    queries_path: str | Path | None = None,
    beams: int = 10,
    max_new_tokens: int = 64,
    device: str = "cuda",
    limit: int | None = None,
    seed: int = 0,
    progress=print,
) -> dict:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    data = Path(data_dir)
    run = Path(run_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    adapter_path = run / "checkpoint-best"
    if not (adapter_path / "adapter_config.json").exists():
        raise FileNotFoundError(f"no adapter at {adapter_path}")

    view = load_toc_view(data / "toc.json")
    test_records, out_of_scope, query_source = load_eval_records(data, queries_path)
    if limit is not None:
        random.Random(seed).shuffle(test_records)
        test_records = test_records[:limit]

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base = AutoModelForCausalLM.from_pretrained(base_model_name, dtype=torch.bfloat16, attn_implementation="sdpa")
    model = PeftModel.from_pretrained(base, str(adapter_path)).to(device).eval()

    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

    probe_prompt = tokenizer.apply_chat_template(
        render_messages(contract, view, test_records[0]["question"]), tokenize=False, add_generation_prompt=True
    )
    trie, sequences = build_leaf_trie(tokenizer, view, probe_prompt)

    # The target tokens must match the trie for the gold leaf of every checked query.
    suffix_checks = 0
    for record in test_records[: min(len(test_records), 25)]:
        prompt = tokenizer.apply_chat_template(
            render_messages(contract, view, record["question"]), tokenize=False, add_generation_prompt=True
        )
        gold_leaf = record["reference"][0]
        layout = training_token_layout(tokenizer, prompt, completion_text([view.label_by_id[gold_leaf]]))
        if layout.target_ids != sequences[gold_leaf]:
            raise AssertionError(f"target tokens for {record.get('query_id')} differ from the trie; refusing to evaluate")
        suffix_checks += 1

    progress(f"Loaded adapter {adapter_path.name}; {trie.leaf_count} leaves in trie; suffix checks={suffix_checks}; queries={len(test_records)} from {query_source}")

    audit = ConstraintAudit()
    per_query: list[dict] = []
    raw_samples: list[dict] = []
    started = time.time()

    for index, record in enumerate(test_records):
        messages = render_messages(contract, view, record["question"])
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        gold = list(record["reference"])

        # The trained input for this question. Every leaf shares it for a given prompt.
        train_ids = training_token_layout(tokenizer, prompt, completion_text([view.label_by_id[gold[0]]])).prefix_ids
        prefix_fn = make_prefix_allowed_tokens_fn(trie, len(train_ids), eos_id, audit)
        upstream_ids = tokenizer(prompt).input_ids
        input_train = torch.tensor([train_ids], device=device)
        input_upstream = torch.tensor([upstream_ids], device=device)
        attention_train = torch.ones_like(input_train)
        attention_upstream = torch.ones_like(input_upstream)

        t0 = time.time()
        with torch.no_grad():
            beam_out = model.generate(
                input_ids=input_train,
                attention_mask=attention_train,
                num_beams=beams,
                num_return_sequences=beams,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                prefix_allowed_tokens_fn=prefix_fn,
                eos_token_id=eos_id,
                pad_token_id=pad_id,
                early_stopping=True,
            )
        constrained_latency = time.time() - t0
        generated = beam_out[:, len(train_ids) :].tolist()
        leaves = sequences_to_leaves(generated, trie, eos_id, audit)
        ranked_constrained: list[str] = []
        for leaf in leaves:
            if leaf is not None and leaf not in ranked_constrained:
                ranked_constrained.append(leaf)
        constrained_items = [
            {"text": view.label_by_id[leaf], "status": "valid_leaf", "node_id": leaf} for leaf in ranked_constrained
        ]

        t1 = time.time()
        with torch.no_grad():
            greedy_train = model.generate(
                input_ids=input_train,
                attention_mask=attention_train,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=eos_id,
                pad_token_id=pad_id,
            )
        unconstrained_latency = time.time() - t1
        raw_train = tokenizer.decode(greedy_train[0][len(train_ids) :], skip_special_tokens=True)

        with torch.no_grad():
            greedy_upstream = model.generate(
                input_ids=input_upstream,
                attention_mask=attention_upstream,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=eos_id,
                pad_token_id=pad_id,
            )
        raw_upstream = tokenizer.decode(greedy_upstream[0][len(upstream_ids) :], skip_special_tokens=True)

        classified_train = classify_output(raw_train, view)
        classified_upstream = classify_output(raw_upstream, view)

        per_query.append(
            {
                "query_id": record.get("query_id"),
                "qtype": record.get("qtype", "unspecified"),
                "question": record["question"],
                "gold": gold,
                "trained_prefix_tokens": len(train_ids),
                "modes": {
                    "constrained": {
                        "ranked": ranked_constrained,
                        "items": constrained_items,
                        "top_status": "valid_leaf" if constrained_items else "unparseable",
                        "latency_seconds": round(constrained_latency, 4),
                    },
                    "unconstrained": {
                        "raw": raw_train,
                        "ranked": list(classified_train.ranked_leaf_ids),
                        "items": [item.__dict__ for item in classified_train.items],
                        "top_status": classified_train.top_status,
                        "latency_seconds": round(unconstrained_latency, 4),
                    },
                    "unconstrained_upstream_prompt": {
                        "raw": raw_upstream,
                        "ranked": list(classified_upstream.ranked_leaf_ids),
                        "items": [item.__dict__ for item in classified_upstream.items],
                        "top_status": classified_upstream.top_status,
                    },
                },
            }
        )
        if index < 3:
            raw_samples.append({"question": record["question"], "constrained": ranked_constrained, "unconstrained_raw": raw_train, "upstream_raw": raw_upstream})
        if (index + 1) % 10 == 0 or index + 1 == len(test_records):
            progress(f"evaluated {index + 1}/{len(test_records)} queries")

    elapsed = time.time() - started
    summaries = _summaries(per_query)
    by_type = {}
    for qtype in sorted({row["qtype"] for row in per_query}):
        by_type[qtype] = _summaries([row for row in per_query if row["qtype"] == qtype])

    with open(out / "per_query.jsonl", "w", encoding="utf-8") as handle:
        for row in per_query:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    result = {
        "evaluated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_dir": str(run),
        "data_dir": str(data),
        "query_source": query_source,
        "out_of_scope_queries": len(out_of_scope),
        "adapter": str(adapter_path),
        "base_model": base_model_name,
        "prompt_contract_sha256": contract.sha256,
        "decoding": {"beams": beams, "max_new_tokens": max_new_tokens, "constrained": "beam search over token trie"},
        "constraint_audit": audit.to_dict(),
        "queries": len(per_query),
        "elapsed_seconds": round(elapsed, 1),
        "summaries": summaries,
        "by_type": by_type,
        "raw_samples": raw_samples,
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result
