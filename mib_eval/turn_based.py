"""Turn-based ablations (paper §6): the Truncated ablation and the Full ablation.

The model is kept from taking the floor until the user audio ends, then gives
one response. The Truncated ablation plays the user audio only through the
first error step (incorrect items only); the Full ablation plays the whole
solution (incorrect and correct items).

For every item directory ``<asr_root>/<id>/`` this module reuses the
``prepare`` parser, takes the model speech after the user's last word as the
response, and asks ``prompts/validity_last_response.txt`` whether it is
Valid: on incorrect items it identifies the error, on correct items it
acknowledges that there is no error. An empty response is ``no_response``,
which is not Valid. Speech during the user's turn is kept for inspection but
not judged.

TP% is the Valid rate on incorrect items. TN% is the Valid rate on correct
items (Full ablation only).
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .prepare import iter_item_dirs, load_annotations, parse_item, user_text_before
from .taxonomy import TURN_BASED_SETTINGS, parse_turn_validity


def _fill(template: str, record: dict) -> str:
    return template.format(
        question=record.get("question") or "",
        ground_truth_solution=record.get("ground_truth_solution") or "",
        user_solution_before_model_speaks=record.get("user_solution_before_model_speaks") or "",
        model_interruption_text=record.get("response_text") or "",
        complete_user_text_intended_to_say=record.get("complete_user_text_intended_to_say") or "",
        error_step_text=record.get("ground_truth_error_step_text"),
        error_reason=record.get("model_output_solution_first_error_reason"),
    )


def parse_turn_item(item_dir: Path, annotations: dict[int, dict]) -> dict:
    record = parse_item(item_dir, annotations)
    if record.get("parse_error"):
        return record
    record["user_solution_before_model_speaks"] = user_text_before(record, None)
    record["response_text"] = record.pop("model_output_after_solution_ends_text")
    record["model_speech_during_user_turn"] = " ".join(u["text"] for u in record.pop("mid_speech_utterances"))
    for key in ("post_hoc_utterances", "_user_solution_chunks", "_input_data"):
        record.pop(key, None)
    return record


def judge_turn_item(record: dict, *, template: str, judge) -> dict:
    """Set ``turn_label`` (valid / invalid / unrelated / no_response), or leave
    it None with an ``unscored_reason``."""
    record["turn_label"] = None
    if record.get("parse_error"):
        record["unscored_reason"] = "parse_failed"
        return record
    if not (record.get("response_text") or "").strip():
        record["turn_label"] = "no_response"
        return record
    if judge is None:
        record["unscored_reason"] = "needs_validity_judge"
        return record
    try:
        record["validity_judgement"] = judge(_fill(template, record))
    except Exception as exc:  # judge failed after its retries
        record["judge_error"] = repr(exc)
        record["unscored_reason"] = "needs_validity_judge"
        return record
    record["turn_label"] = parse_turn_validity(record["validity_judgement"])
    if record["turn_label"] is None:
        record["unscored_reason"] = "unreadable_judgement"
    return record


def _pct(num: int, den: int):
    return round(100.0 * num / den, 1) if den else None


def summarize_turn_based(records: list[dict], setting: str, allow_unscored: bool = False) -> dict:
    """Label counts per half, plus TP% and (Full ablation) TN%.

    Rates are over scored items. If any item is unscored, the metrics are
    None unless ``allow_unscored`` is set.
    """
    halves = {"incorrect": {}, "correct": {}}
    unscored: dict[str, int] = {}
    for r in records:
        if r["turn_label"] is None:
            reason = r.get("unscored_reason") or "unscored"
            unscored[reason] = unscored.get(reason, 0) + 1
            continue
        half = halves["incorrect" if r.get("has_error") else "correct"]
        half[r["turn_label"]] = half.get(r["turn_label"], 0) + 1

    summary = {
        "setting": TURN_BASED_SETTINGS[setting],
        "n": len(records),
        "labels": halves,
        "unscored": unscored,
        "metrics": None,
    }
    if unscored and not allow_unscored:
        return summary

    n_inc = sum(halves["incorrect"].values())
    n_cor = sum(halves["correct"].values())
    metrics = {"TP%": _pct(halves["incorrect"].get("valid", 0), n_inc)}
    if setting == "full":
        metrics["TN%"] = _pct(halves["correct"].get("valid", 0), n_cor)
    summary["metrics"] = metrics
    return summary


def score_turn_based(asr_root: Path, annotations_path: Path, output_path: Path, *, setting: str,
                     judge, template_path: Path, workers: int = 8, allow_unscored: bool = False) -> dict:
    if setting not in TURN_BASED_SETTINGS:
        raise ValueError(f"setting must be one of {sorted(TURN_BASED_SETTINGS)}; got {setting!r}")
    if output_path.exists():
        raise ValueError(f"{output_path} already exists. Pass a new --output.")

    annotations = load_annotations(annotations_path)
    records = [parse_turn_item(d, annotations) for d in iter_item_dirs(asr_root)]
    excluded = 0
    if setting == "truncated":
        # The Truncated ablation is defined on incorrect items only.
        kept = [r for r in records if r.get("has_error") or r.get("has_error") is None]
        excluded = len(records) - len(kept)
        records = kept

    template = template_path.read_text()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        records = list(pool.map(lambda r: judge_turn_item(r, template=template, judge=judge), records))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, indent=2, ensure_ascii=False))
    summary = summarize_turn_based(records, setting, allow_unscored=allow_unscored)
    summary["excluded_correct_items"] = excluded
    return summary
