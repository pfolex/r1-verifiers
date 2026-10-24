"""Command line interface for r1-verifiers (`r1-verify`)."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from r1_verifiers.grpo import LengthLeadReward
from r1_verifiers.pipeline import ReasoningVerificationPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="r1-verify",
        description="Verify and score reasoning model completions (DeepSeek-R1, QwQ, etc.).",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--completion",
        type=str,
        help="Single model completion text including <think> tags and boxed answer.",
    )
    group.add_argument(
        "--file",
        type=str,
        help="Path to JSON file containing a list of rollout objects with 'completion' and 'gold'.",
    )

    parser.add_argument(
        "--gold",
        type=str,
        default="",
        help="Gold reference answer (required when --completion is passed).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="GRPO-LEAD length penalty parameter (default: 0.5).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON instead of human-readable summary.",
    )
    return parser.parse_args()


def run_single(completion: str, gold: str, as_json: bool) -> int:
    pipeline = ReasoningVerificationPipeline()
    report = pipeline.verify(completion, gold)

    if as_json:
        data = {
            "is_correct": report.is_correct,
            "composite_score": report.composite_score,
            "format": {
                "is_valid": report.format_result.is_valid,
                "score": report.format_result.score,
                "extracted_answer": report.format_result.extracted_answer,
                "issues": report.format_result.issues,
            },
            "math": {
                "is_correct": report.math_result.is_correct,
                "score": report.math_result.score,
                "gold": report.math_result.gold_answer,
                "extracted": report.math_result.extracted_answer,
            },
            "reflection": {
                "reflections": report.reflection_result.reflection_count,
                "verification_step": report.reflection_result.has_verification_step,
                "score": report.reflection_result.score,
                "detected_markers": report.reflection_result.detected_markers,
            },
            "token_estimates": {
                "thought_tokens": report.thought_tokens,
                "solution_tokens": report.solution_tokens,
            },
        }
        print(json.dumps(data, indent=2))
    else:
        status = "PASSED" if report.is_correct else "FAILED"
        print("=" * 60)
        print(f"  R1 REASONING VERIFIER: [{status}] (Score: {report.composite_score:.2f})")
        print("=" * 60)
        print(f"Extracted Answer : {report.format_result.extracted_answer}")
        print(f"Gold Reference   : {report.math_result.gold_answer}")
        print(f"Format Valid     : {report.format_result.is_valid} ({report.format_result.score:.2f})")
        if report.format_result.issues:
            print(f"  Issues         : {', '.join(report.format_result.issues)}")
        print(f"Reflections Found: {report.reflection_result.reflection_count} ({report.reflection_result.score:.2f})")
        if report.reflection_result.detected_markers:
            print(f"  Markers        : {', '.join(report.reflection_result.detected_markers[:5])}")
        print(f"Thought Length   : ~{report.thought_tokens} tokens")
        print("=" * 60)

    return 0 if report.is_correct else 1


def run_batch(file_path: str, alpha: float, as_json: bool) -> int:
    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    default_gold = ""
    if isinstance(raw_data, dict):
        default_gold = str(raw_data.get("gold") or raw_data.get("solution") or raw_data.get("target") or "")
        items = raw_data.get("rollouts") or raw_data.get("items") or [raw_data]
    elif isinstance(raw_data, list):
        items = raw_data
    else:
        items = []

    pipeline = ReasoningVerificationPipeline()
    reports = []
    correctness = []
    lengths = []

    for item in items:
        if isinstance(item, str):
            comp = item
            gold = default_gold
        else:
            comp = (
                item.get("full_completion")
                or item.get("completion")
                or item.get("solution")
                or item.get("text", "")
            )
            gold = (
                item.get("gold")
                or item.get("gold_answer")
                or item.get("target")
                or default_gold
            )
        rep = pipeline.verify(comp, gold)
        reports.append((item, rep))
        correctness.append(rep.is_correct)
        lengths.append(rep.thought_tokens + rep.solution_tokens)

    grpo_scorer = LengthLeadReward(alpha=alpha)
    grpo_res = grpo_scorer.score_group(correctness, lengths)

    if as_json:
        output_list = []
        for (_item, rep), grpo_score in zip(reports, grpo_res.scores, strict=False):
            output_list.append({
                "index": grpo_score.index,
                "is_correct": rep.is_correct,
                "composite_score": rep.composite_score,
                "length": grpo_score.length,
                "length_z_score": grpo_score.length_z_score,
                "raw_reward": grpo_score.raw_reward,
                "advantage": grpo_score.advantage,
                "extracted_answer": rep.format_result.extracted_answer,
            })
        print(json.dumps({
            "group_stats": {
                "mean_length": grpo_res.mean_length,
                "std_length": grpo_res.std_length,
                "mean_reward": grpo_res.mean_reward,
            },
            "rollouts": output_list,
        }, indent=2))
    else:
        print("=" * 70)
        print(f"  BATCH VERIFICATION & GRPO-LEAD ADVANTAGE EVALUATION (N={len(items)})")
        print("=" * 70)
        print(f"{'Idx':<4} | {'Correct':<7} | {'Tokens':<6} | {'z_len':<7} | {'Reward':<8} | {'Advantage':<9} | {'Answer'}")
        print("-" * 70)
        for (_, rep), g_score in zip(reports, grpo_res.scores, strict=False):
            corr_str = "YES" if rep.is_correct else "NO"
            ans = str(rep.format_result.extracted_answer)
            print(
                f"{g_score.index:<4} | {corr_str:<7} | {g_score.length:<6} | {g_score.length_z_score:>+6.2f} | "
                f"{g_score.raw_reward:>+7.3f} | {g_score.advantage:>+8.3f} | {ans}"
            )
        print("=" * 70)

    return 0


def main() -> None:
    args = parse_args()
    if args.completion is not None:
        if not args.gold:
            print("Error: --gold reference answer is required when using --completion", file=sys.stderr)
            sys.exit(2)
        code = run_single(args.completion, args.gold, args.json)
        sys.exit(code)
    elif args.file is not None:
        code = run_batch(args.file, args.alpha, args.json)
        sys.exit(code)


if __name__ == "__main__":
    main()
