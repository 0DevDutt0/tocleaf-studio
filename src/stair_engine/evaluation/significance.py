"""Paired comparison of two systems on the same queries: an exact McNemar test on top-1 correctness.

A query is correct when the first ranked leaf is one of its gold leaves. Only the queries where the two systems disagree
carry information. The exact binomial test uses those discordant counts and makes no large-sample assumption, which
matters with 47 test queries.

Command line, from the project root:
    uv run python -m stair_engine.evaluation.significance A_per_query.jsonl B_per_query.jsonl --mode constrained
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def top1_correct(path: str | Path, mode: str | None = None) -> dict[str, bool]:
    """Map each query id to whether its first ranked leaf is gold.

    STAIR per-query files keep one ranking per decoding mode under ``modes``, so a mode is required for them. BM25 files
    have a single ``ranked`` list, and the mode is ignored for them.
    """

    results: dict[str, bool] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "modes" in row:
            if mode is None:
                raise ValueError(f"{path} holds several decoding modes; name the one to compare")
            ranked = row["modes"][mode]["ranked"]
        else:
            ranked = row["ranked"]
        gold = row["gold"] if isinstance(row["gold"], list) else [row["gold"]]
        results[str(row["query_id"])] = bool(ranked) and ranked[0] in set(gold)
    return results


def exact_mcnemar_two_sided(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar p-value from the two discordant counts."""

    discordant = only_a + only_b
    if discordant == 0:
        return 1.0
    smaller = min(only_a, only_b)
    tail = sum(math.comb(discordant, k) for k in range(smaller + 1)) / 2**discordant
    return min(1.0, 2.0 * tail)


def compare(path_a: str | Path, path_b: str | Path, mode: str | None = None) -> dict:
    """Compare system A with system B on the queries they share, which must be the same set."""

    a = top1_correct(path_a, mode)
    b = top1_correct(path_b, mode)
    if set(a) != set(b):
        raise ValueError(f"query sets differ: {len(a)} queries against {len(b)}")
    ids = sorted(a)
    only_a = sum(1 for query in ids if a[query] and not b[query])
    only_b = sum(1 for query in ids if b[query] and not a[query])
    return {
        "mode": mode,
        "queries": len(ids),
        "top1_A": round(sum(a.values()) / len(ids), 4),
        "top1_B": round(sum(b.values()) / len(ids), 4),
        "A_right_B_wrong": only_a,
        "A_wrong_B_right": only_b,
        "net_queries_A_minus_B": only_a - only_b,
        "mcnemar_exact_p_two_sided": round(exact_mcnemar_two_sided(only_a, only_b), 5),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Exact McNemar test on top-1 correctness of two per-query files.")
    parser.add_argument("a", help="per-query file of system A")
    parser.add_argument("b", help="per-query file of system B")
    parser.add_argument("--mode", default=None, help="decoding mode for STAIR files, for example constrained")
    args = parser.parse_args(argv)
    report = compare(args.a, args.b, args.mode)
    # Print only the folder and file names, so the output carries no home-directory path.
    report["A"] = f"{Path(args.a).parent.name}/{Path(args.a).name}"
    report["B"] = f"{Path(args.b).parent.name}/{Path(args.b).name}"
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
