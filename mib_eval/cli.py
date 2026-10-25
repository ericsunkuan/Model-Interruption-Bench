"""CLI: python -m mib_eval {prepare-full-duplex,score-full-duplex,score-turn-based} ..."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .judge import make_judge
from .prepare import prepare_file
from .score import score_streaming_file
from .taxonomy import TURN_BASED_SETTINGS
from .turn_based import score_turn_based

ROOT = Path(__file__).resolve().parents[1]
VALIDITY_PROMPT = ROOT / "prompts" / "validity.txt"
FOUR_WAY_PROMPT = ROOT / "prompts" / "four_way.txt"
TIMESTAMP_PROMPT = ROOT / "prompts" / "timestamp_mapping_evaluation_prompt.txt"
BACKCHANNEL_PROMPT = ROOT / "prompts" / "interruption_type.txt"
TURN_BASED_PROMPT = ROOT / "prompts" / "validity_last_response.txt"
DEFAULT_CACHE = ROOT / ".openai_cache" / "judge_cache.jsonl"


def _add_judge_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--model", default="gpt-5.2-2025-12-11")
    p.add_argument("--workers", type=int, default=8, help="Parallel judge calls")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MIB full-duplex evaluation")
    sub = parser.add_subparsers(dest="cmd", required=True)

    prep = sub.add_parser(
        "prepare-full-duplex",
        help="Parse ASR transcripts, map the error-step start, and run the backchannel judge",
    )
    prep.add_argument("--asr-root", type=Path, required=True, help=".../<model>/gsm8k_full_duplex with <id>/input.json, output.json")
    prep.add_argument("--annotations", type=Path, required=True, help="GSM8K annotation JSON")
    prep.add_argument("--output", type=Path, required=True)
    _add_judge_args(prep)

    score = sub.add_parser("score-full-duplex", help="Score a full-duplex result JSON")
    score.add_argument("--results", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument(
        "--judge", "--judge-posthoc", dest="judge", action="store_true",
        help="Call the LLM judge for missing in-window validity, post-hoc validity, and the 4-way label",
    )
    score.add_argument(
        "--allow-unscored", action="store_true",
        help="Compute metrics over the scored items even if some items are unscored",
    )
    _add_judge_args(score)

    turn = sub.add_parser(
        "score-turn-based",
        help="Score the Truncated ablation or the Full ablation (one response after the user finishes)",
    )
    turn.add_argument("--setting", choices=sorted(TURN_BASED_SETTINGS), required=True,
                      help="truncated = Truncated ablation, full = Full ablation")
    turn.add_argument("--asr-root", type=Path, required=True, help=".../<model>/gsm8k_<setting> with <id>/input.json, output.json")
    turn.add_argument("--annotations", type=Path, required=True, help="GSM8K annotation JSON")
    turn.add_argument("--output", type=Path, required=True)
    turn.add_argument("--allow-unscored", action="store_true",
                      help="Compute metrics over the scored items even if some items are unscored")
    _add_judge_args(turn)

    args = parser.parse_args(argv)
    if args.cmd == "score-turn-based":
        summary = score_turn_based(
            args.asr_root,
            args.annotations,
            args.output,
            setting=args.setting,
            judge=make_judge(args.cache, model=args.model),
            template_path=TURN_BASED_PROMPT,
            workers=args.workers,
            allow_unscored=args.allow_unscored,
        )
        print(json.dumps(summary, indent=2))
        return _report_unscored(summary, args.allow_unscored)

    if args.cmd == "prepare-full-duplex":
        report = prepare_file(
            args.asr_root,
            args.annotations,
            args.output,
            judge=make_judge(args.cache, model=args.model),
            timestamp_path=TIMESTAMP_PROMPT,
            backchannel_path=BACKCHANNEL_PROMPT,
            workers=args.workers,
        )
        print(json.dumps(report, indent=2))
        if report["parse_errors"] or report["judge_errors"]:
            print(
                f"WARNING: {len(report['parse_errors'])} items could not be parsed and "
                f"{len(report['judge_errors'])} had judge failures. They are kept in the "
                "output and will be reported as unscored.",
                file=sys.stderr,
            )
        return 0

    if args.cmd == "score-full-duplex":
        judge = make_judge(args.cache, model=args.model) if args.judge else None
        summary = score_streaming_file(
            args.results,
            args.output,
            prompt_path=VALIDITY_PROMPT,
            four_way_path=FOUR_WAY_PROMPT,
            judge=judge,
            workers=args.workers,
            allow_unscored=args.allow_unscored,
        )
        print(json.dumps(summary, indent=2))
        return _report_unscored(summary, args.allow_unscored)
    return 0


def _report_unscored(summary: dict, allow_unscored: bool) -> int:
    if not summary["unscored"]:
        return 0
    n = sum(summary["unscored"].values())
    if allow_unscored:
        print(f"WARNING: {n} items are unscored ({summary['unscored']}). "
              "Metrics cover the scored items only.", file=sys.stderr)
        return 0
    print(
        f"ERROR: {n} items are unscored ({summary['unscored']}). Metrics were not "
        "computed. Rerun with --judge, check the prepare step, or pass --allow-unscored.",
        file=sys.stderr,
    )
    return 1
