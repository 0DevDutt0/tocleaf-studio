"""Local LoRA trainer that selects checkpoints by the paper's rule: constrained Recall@1 on the dev set.

The pinned upstream trainer selects by evaluation loss, and it runs a text-generation callback at every evaluation.
This trainer follows the paper instead (Section 5.2): it evaluates Recall@1 on the dev split, uses the constrained
ranking that the evaluator uses, stops when that Recall@1 has not improved for `patience_epochs`, and keeps the best
adapter. It also keeps the lowest-dev-loss adapter from the same trajectory, so the two selection rules can be compared
on identical training. The paper uses up to 200 epochs and patience 20. Those numbers are scaled down here, and the run manifest
records the values used.

Training uses the same prompt contract and token layout as the upstream trainer, so the adapter loads with the same
evaluator, and the loss is taken on the completion tokens only (the leaf label and the end-of-turn token).
"""

from __future__ import annotations

import datetime as dt
import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

from stair_engine.inference.constrained import TokenTrie
from stair_engine.inference.contract import (
    PromptContract,
    ToCView,
    completion_text,
    load_toc_view,
    render_messages,
    training_token_layout,
)

DEFAULT_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj")


@dataclass
class LocalTrainingConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: tuple[str, ...] = DEFAULT_TARGET_MODULES
    learning_rate: float = 1e-4
    warmup_steps: int = 20
    weight_decay: float = 0.05
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-10
    micro_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    max_epochs: int = 20
    patience_epochs: int = 5
    seed: int = 42
    max_seq_length: int = 2048
    extra: dict = field(default_factory=dict)


@dataclass
class TrainingExample:
    query_id: str
    prefix: list[int]  # what the trainer saw before the first completion token
    target: list[int]  # completion tokens, ending with end-of-turn
    gold_leaf: str
    question: str


def build_examples(tokenizer, view: ToCView, contract: PromptContract, records: list[dict]) -> list[TrainingExample]:
    examples: list[TrainingExample] = []
    for record in records:
        leaf = record["reference"][0]
        prompt = tokenizer.apply_chat_template(
            render_messages(contract, view, record["question"]), tokenize=False, add_generation_prompt=True
        )
        layout = training_token_layout(tokenizer, prompt, completion_text([view.label_by_id[leaf]]))
        if len(layout.prefix_ids) + len(layout.target_ids) > 2048:
            raise ValueError(f"{record.get('query_id')} is longer than 2048 tokens; refusing to train on it")
        examples.append(
            TrainingExample(
                query_id=str(record.get("query_id")),
                prefix=list(layout.prefix_ids),
                target=list(layout.target_ids),
                gold_leaf=leaf,
                question=record["question"],
            )
        )
    return examples


def batch_nll(core, lm_head, batch: list[TrainingExample], pad_id: int, device: str) -> torch.Tensor:
    """Mean negative log-likelihood of the completion tokens over the batch (token-level mean).

    ``core`` is the decoder stack and ``lm_head`` the output projection. The completion of each example is predicted
    from the positions just before it: the input is the full sequence without its last token, and the targets are the
    completion tokens, so the loss is taken on completion tokens only.
    """

    longest = max(len(example.prefix) + len(example.target) - 1 for example in batch)
    input_ids = torch.full((len(batch), longest), pad_id, dtype=torch.long, device=device)
    attention = torch.zeros((len(batch), longest), dtype=torch.long, device=device)
    for row, example in enumerate(batch):
        sequence = example.prefix + example.target
        inputs = sequence[:-1]
        input_ids[row, : len(inputs)] = torch.tensor(inputs, device=device)
        attention[row, : len(inputs)] = 1
    hidden = core(input_ids=input_ids, attention_mask=attention).last_hidden_state
    total = hidden.new_zeros((), dtype=torch.float32)
    tokens = 0
    for row, example in enumerate(batch):
        start = len(example.prefix) - 1
        count = len(example.target)
        logits = lm_head(hidden[row, start : start + count]).float()
        targets = torch.tensor(example.target, device=device)
        total = total + F.cross_entropy(logits, targets, reduction="sum")
        tokens += count
    return total / tokens


def restricted_leaf_scores(
    core,
    lm_head,
    prefix: list[int],
    sequences: dict[str, list[int]],
    trie: TokenTrie,
    device: str,
) -> dict[str, float]:
    """Constrained log-probability of every valid leaf, as the constrained decoder would score it.

    Each step is renormalised over the trie's valid continuations. All candidates share one batched forward pass.
    """

    leaves = list(sequences)
    longest = max(len(prefix) + len(sequences[leaf]) - 1 for leaf in leaves)
    input_ids = torch.full((len(leaves), longest), 0, dtype=torch.long, device=device)
    attention = torch.zeros((len(leaves), longest), dtype=torch.long, device=device)
    for row, leaf in enumerate(leaves):
        sequence = prefix + sequences[leaf]
        inputs = sequence[:-1]
        input_ids[row, : len(inputs)] = torch.tensor(inputs, device=device)
        attention[row, : len(inputs)] = 1
    with torch.no_grad():
        hidden = core(input_ids=input_ids, attention_mask=attention).last_hidden_state
        scores: dict[str, float] = {}
        for row, leaf in enumerate(leaves):
            target = sequences[leaf]
            start = len(prefix) - 1
            logits = lm_head(hidden[row, start : start + len(target)]).float()
            total = 0.0
            for step, token in enumerate(target):
                allowed = trie.allowed_next(tuple(target[:step]))
                allowed_logits = logits[step][torch.tensor(allowed, device=device)]
                total += float(F.log_softmax(allowed_logits, dim=-1)[allowed.index(token)])
            scores[leaf] = total
    return scores


def dev_recall_at_1(core, lm_head, examples: list[TrainingExample], sequences, trie, device: str) -> float:
    hits = 0
    for example in examples:
        scores = restricted_leaf_scores(core, lm_head, example.prefix, sequences, trie, device)
        top = max(scores, key=lambda leaf: (scores[leaf], leaf))
        hits += top == example.gold_leaf
    return hits / max(1, len(examples))


def run_local_training(
    *,
    data_dir: str | Path,
    run_dir: str | Path,
    contract: PromptContract,
    config: LocalTrainingConfig,
    device: str = "cuda",
    progress=print,
) -> dict:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from stair_engine.evaluation.stair_runner import build_leaf_trie

    data = Path(data_dir)
    run = Path(run_dir)
    if run.exists() and any(run.glob("checkpoint-*")):
        raise FileExistsError(f"{run} already holds checkpoints; choose a fresh run directory")
    run.mkdir(parents=True, exist_ok=True)

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    view = load_toc_view(data / "toc.json")
    train_rows = [json.loads(line) for line in (data / "train.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    dev_rows = [json.loads(line) for line in (data / "val.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    train_examples = build_examples(tokenizer, view, contract, train_rows)
    dev_examples = build_examples(tokenizer, view, contract, dev_rows)
    probe = tokenizer.apply_chat_template(
        render_messages(contract, view, train_examples[0].question), tokenize=False, add_generation_prompt=True
    )
    trie, sequences = build_leaf_trie(tokenizer, view, probe)

    base = AutoModelForCausalLM.from_pretrained(config.model_name, dtype=torch.bfloat16, attn_implementation="sdpa").to(device)
    base.config.use_cache = False
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base.enable_input_require_grads()
    lora = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora)
    decoder = model.get_base_model()
    core = decoder.model
    lm_head = decoder.lm_head
    trainable = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=config.learning_rate, betas=config.betas, eps=config.eps, weight_decay=config.weight_decay
    )

    def learning_rate_factor(step: int) -> float:
        return min(1.0, (step + 1) / max(1, config.warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_factor)
    progress(
        f"train {len(train_examples)} examples, dev {len(dev_examples)}, leaves {len(sequences)}; "
        f"trainable parameters {sum(p.numel() for p in trainable):,}"
    )

    history: list[dict] = []
    best = {"dev_recall_at_1": -1.0, "dev_loss": float("inf"), "epoch": 0}
    best_epoch_index = -1
    best_by_loss = {"dev_loss": float("inf"), "dev_recall_at_1": None, "epoch": 0}
    started = time.time()
    torch.cuda.reset_peak_memory_stats() if device.startswith("cuda") else None
    order_rng = random.Random(config.seed)

    for epoch in range(config.max_epochs):
        model.train()
        order = list(train_examples)
        order_rng.shuffle(order)
        step_losses: list[float] = []
        micro = config.micro_batch_size
        accum = config.gradient_accumulation_steps
        batches = [order[index : index + micro] for index in range(0, len(order), micro)]
        for start in range(0, len(batches), accum):
            group = batches[start : start + accum]
            for batch in group:
                loss = batch_nll(core, lm_head, batch, pad_id, device) / len(group)
                loss.backward()
                step_losses.append(float(loss.detach()) * len(group))
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        model.eval()
        with torch.no_grad():
            dev_loss = sum(
                float(batch_nll(core, lm_head, dev_examples[index : index + micro], pad_id, device))
                * len(dev_examples[index : index + micro])
                for index in range(0, len(dev_examples), micro)
            ) / max(1, len(dev_examples))
        dev_recall = dev_recall_at_1(core, lm_head, dev_examples, sequences, trie, device)
        entry = {
            "epoch": epoch + 1,
            "optimizer_steps": scheduler.last_epoch,
            "train_loss": round(sum(step_losses) / max(1, len(step_losses)), 6),
            "dev_loss": round(dev_loss, 6),
            "dev_recall_at_1": round(dev_recall, 6),
        }
        history.append(entry)
        progress(f"epoch {epoch + 1}: train_loss {entry['train_loss']:.4f} dev_loss {entry['dev_loss']:.4f} dev_R@1 {entry['dev_recall_at_1']:.3f}")

        if dev_loss < best_by_loss["dev_loss"]:
            # A control: the same trajectory, selected by dev loss instead of dev Recall@1.
            best_by_loss = {"dev_loss": dev_loss, "dev_recall_at_1": dev_recall, "epoch": epoch + 1}
            model.save_pretrained(run / "checkpoint-best-loss")

        improved = dev_recall > best["dev_recall_at_1"] or (
            dev_recall == best["dev_recall_at_1"] and dev_loss < best["dev_loss"]
        )
        if improved:
            best = {"dev_recall_at_1": dev_recall, "dev_loss": dev_loss, "epoch": epoch + 1}
            best_epoch_index = epoch
            model.save_pretrained(run / "checkpoint-best")
        if epoch - best_epoch_index >= config.patience_epochs:
            progress(f"stopping: no dev Recall@1 improvement for {config.patience_epochs} epochs")
            break

    manifest = {
        "trainer": "tocleaf local trainer (paper's rule: constrained dev Recall@1, patience in epochs)",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "data_dir": str(data),
        "run_dir": str(run),
        "config": asdict(config),
        "prompt_contract_sha256": contract.sha256,
        "best": best,
        "history": history,
        "examples": {"train": len(train_examples), "dev": len(dev_examples), "leaves": len(sequences)},
        "duration_seconds": round(time.time() - started, 1),
        "peak_gpu_memory_mib": (
            round(torch.cuda.max_memory_allocated() / 2**20) if device.startswith("cuda") else None
        ),
        "adapter_saved": (run / "checkpoint-best" / "adapter_config.json").exists(),
        "best_by_dev_loss": best_by_loss,
        "adapter_saved_by_dev_loss": (run / "checkpoint-best-loss" / "adapter_config.json").exists(),
    }
    (run / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
