"""Apply the paper full-duplex taxonomy to an existing result file.

Reads a result JSON produced by ``prepare-full-duplex`` (backchannel labels,
error-step start). Writes a new JSON. Does not modify the input file.

Incorrect audio:
  The first non-backchannel utterance is placed against [t_s, t_e + 10].
  In-window interruptions get the 3-way validity label (judged here if the
  input does not already carry one); non-Valid ones are FN ``wrong_error``.
  When there is no mid-speech non-backchannel utterance, a non-empty
  post-user-end transcript is labeled ``ph_corr`` or ``ph_other`` from the
  same 3-way validity prompt used for in-window interruptions.

Correct audio (paper Table 8):
  The first mid-speech non-backchannel utterance gets the 4-way label
  (prompts/four_way.txt). (B) affirm, (C) false_claim, (D) unrelated are FP.
  On (A), a non-empty post-user-end transcript gets the 3-way validity label
  with the error fields set to None. Invalid is FP ``ph_claim``; otherwise TN.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

from .taxonomy import (
    assign_streaming_correct,
    assign_streaming_error,
    interruption_latency,
    parse_four_way,
    parse_streaming_validity,
    parse_timestamp,
)


def _float_or_none(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fill_validity(template: str, item: dict, assistant_text: str) -> str:
    return template.format(
        question=item.get("question") or "",
        ground_truth_solution=item.get("ground_truth_solution") or "",
        model_interruption_text=assistant_text,
        complete_user_text_intended_to_say=item.get("complete_user_text_intended_to_say") or "",
        error_step_text=item.get("ground_truth_error_step_text"),
        error_reason=item.get("model_output_solution_first_error_reason"),
    )


def _fill_four_way(template: str, item: dict) -> str:
    return template.format(
        question=item.get("question") or "",
        user_solution_before_model_speaks=item.get("user_solution_before_model_speaks") or "",
        model_interruption_text=item.get("first_non_bc_interruption_text") or "",
    )


def _posthoc_validity(out: dict, post_text: str, validity_template: str | None, judge) -> str | None:
    """Judge the post-user-end text, or reuse a stored label when no judge is given."""
    if judge is not None and validity_template:
        raw = judge(_fill_validity(validity_template, out, post_text))
        out["posthoc_validity_judgement"] = raw
    return parse_streaming_validity(out.get("posthoc_validity_judgement"))


def score_error_item(item: dict, *, validity_template: str | None, judge) -> dict:
    out = deepcopy(item)
    # The timestamp prompt answers -1 when it cannot find the step; that is
    # not a time, so it becomes None and the item cannot be placed in a window.
    start = parse_timestamp(out.get("ground_truth_error_step_timestamp"))
    end = _float_or_none(out.get("ground_truth_error_step_end_timestamp"))

    has_non_bc = out.get("has_any_non_bc")
    post_text = out.get("model_output_after_solution_ends_text") or ""
    fields = dict(
        has_mid_speech_non_bc=None if has_non_bc is None else bool(has_non_bc),
        first_non_bc_time=_float_or_none(out.get("first_non_bc_interruption_time")),
        error_start=start,
        error_end=end,
        posthoc_text=post_text,
    )
    in_window = parse_streaming_validity(out.get("validity_judgement"))
    posthoc = parse_streaming_validity(out.get("posthoc_validity_judgement"))
    preview = assign_streaming_error(**fields, in_window_validity=in_window, posthoc_validity=posthoc)
    if judge is not None and validity_template:
        if preview["fn_sub_type"] == "needs_validity_judge":
            out["validity_judgement"] = judge(_fill_validity(
                validity_template, out, out.get("first_non_bc_interruption_text") or ""))
            in_window = parse_streaming_validity(out["validity_judgement"])
        elif preview["fn_sub_type"] in ("needs_posthoc_judge", "ph_corr", "ph_other"):
            posthoc = _posthoc_validity(out, post_text, validity_template, judge)
    out.update(assign_streaming_error(**fields, in_window_validity=in_window, posthoc_validity=posthoc))
    out["latency_s"] = interruption_latency(
        fields["first_non_bc_time"] if fields["has_mid_speech_non_bc"] else None,
        start, end, _float_or_none(out.get("user_solution_end_time")),
    )
    return out


def score_correct_item(item: dict, *, validity_template: str | None,
                       four_way_template: str | None, judge) -> dict:
    out = deepcopy(item)
    has_non_bc = out.get("has_any_non_bc")
    if has_non_bc is not None:
        has_non_bc = bool(has_non_bc)

    four_way = None
    if has_non_bc is False:
        four_way = "A"
        out["four_way_judgement"] = None
        out["four_way_label"] = "A"
    elif has_non_bc:
        if judge is not None and four_way_template:
            out["four_way_judgement"] = judge(_fill_four_way(four_way_template, out))
        four_way = parse_four_way(out.get("four_way_judgement"))
        out["four_way_label"] = four_way

    post_text = out.get("model_output_after_solution_ends_text") or ""
    posthoc = None
    if four_way == "A" and post_text.strip():
        posthoc = _posthoc_validity(out, post_text, validity_template, judge)

    out.update(
        assign_streaming_correct(
            has_mid_speech_non_bc=has_non_bc,
            four_way=four_way,
            posthoc_text=post_text,
            posthoc_validity=posthoc,
        )
    )
    return out


def score_streaming_item(
    item: dict,
    *,
    prompt_template: str | None,
    judge,
    four_way_template: str | None = None,
) -> dict:
    if item.get("parse_error"):
        # prepare-full-duplex could not read this item. Keep it visible as
        # unscored instead of dropping it from the denominators.
        out = deepcopy(item)
        side = "fn_sub_type" if item.get("has_error") else "fp_sub_type"
        out.update({"result_type": None, "fn_sub_type": None, "fp_sub_type": None})
        out[side] = "parse_failed"
        return out
    try:
        if item.get("has_error"):
            return score_error_item(item, validity_template=prompt_template, judge=judge)
        return score_correct_item(
            item, validity_template=prompt_template,
            four_way_template=four_way_template, judge=judge,
        )
    except RuntimeError as exc:  # a judge call failed after its retries
        out = deepcopy(item)
        side = "fn_sub_type" if item.get("has_error") else "fp_sub_type"
        out.update({"result_type": None, "fn_sub_type": None, "fp_sub_type": None,
                    "judge_error": repr(exc)})
        out[side] = "judge_failed"
        return out


def score_streaming_file(
    results_path: Path,
    output_path: Path,
    *,
    prompt_path: Path | None = None,
    four_way_path: Path | None = None,
    judge=None,
    workers: int = 8,
    allow_unscored: bool = False,
) -> dict:
    if output_path.resolve() == results_path.resolve():
        raise ValueError("Refusing to overwrite the input results file. Pass a different --output.")

    items = json.loads(results_path.read_text())
    template = prompt_path.read_text() if prompt_path else None
    four_way_template = four_way_path.read_text() if four_way_path else None

    def run(item):
        return score_streaming_item(
            item,
            prompt_template=template,
            judge=judge,
            four_way_template=four_way_template,
        )

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        scored = list(pool.map(run, items))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(scored, indent=2, ensure_ascii=False))
    return summarize(scored, allow_unscored=allow_unscored)


def _pct(num: int, den: int):
    return round(100.0 * num / den, 1) if den else None


def summarize(items: list[dict], allow_unscored: bool = False) -> dict:
    """Counts, sub-types and paper metrics.

    Rates follow the paper: TP%/FN% over error items, TN%/FP% over correct
    items, precision = TP/(TP+FP), recall = TP/(TP+FN). Latency is the mean
    of ``latency_s`` over scored items that have one. If any item is
    unscored, the metrics are None so a partial run is never reported,
    unless ``allow_unscored`` is set: then the rates are computed over the
    scored items only and ``unscored`` says how many were left out.
    """
    counts = {"TP": 0, "FN": 0, "TN": 0, "FP": 0}
    fn: dict[str, int] = {}
    fp: dict[str, int] = {}
    unscored: dict[str, int] = {}
    latencies: list[float] = []
    n_error = n_correct = n_unknown = 0
    for item in items:
        if item.get("has_error") is None:
            n_unknown += 1  # no annotation could be read
        elif item["has_error"]:
            n_error += 1
        else:
            n_correct += 1
        label = item.get("result_type")
        if label not in counts:
            reason = item.get("fn_sub_type") or item.get("fp_sub_type") or "missing_result_type"
            unscored[reason] = unscored.get(reason, 0) + 1
            continue
        counts[label] += 1
        if item.get("latency_s") is not None:
            latencies.append(float(item["latency_s"]))
        if label == "FN":
            sub = item.get("fn_sub_type") or "unknown"
            fn[sub] = fn.get(sub, 0) + 1
        if label == "FP":
            sub = item.get("fp_sub_type") or "unknown"
            fp[sub] = fp.get(sub, 0) + 1

    summary = {
        "n": len(items),
        "n_error": n_error,
        "n_correct": n_correct,
        "n_unknown": n_unknown,
        "counts": counts,
        "fn_sub_types": fn,
        "fp_sub_types": fp,
        "unscored": unscored,
        "metrics": None,
        "latency": None,
    }
    if unscored and not allow_unscored:
        return summary

    # Paper §4.1: mean over incorrect-audio items whose first non-backchannel
    # interruption falls in [t_s, t_u].
    summary["latency"] = {
        "mean_s": round(sum(latencies) / len(latencies), 2) if latencies else None,
        "n": len(latencies),
    }

    tp, fn_n, tn, fp_n = counts["TP"], counts["FN"], counts["TN"], counts["FP"]
    n_error, n_correct = tp + fn_n, tn + fp_n  # scored items only
    precision = tp / (tp + fp_n) if tp + fp_n else 0.0
    recall = tp / (tp + fn_n) if tp + fn_n else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    summary["metrics"] = {
        "TP%": _pct(tp, n_error),
        "FN%": _pct(fn_n, n_error),
        "TN%": _pct(tn, n_correct),
        "FP%": _pct(fp_n, n_correct),
        "P%": round(100 * precision, 2),
        "R%": round(100 * recall, 2),
        "F1%": round(100 * f1, 2),
    }
    return summary
