"""Paper TP/FN/TN/FP assignment for the full-duplex evaluation.

Incorrect audio: the window is [t_s, t_e + 10]. In-window content that is
not Valid is stored as FN ``wrong_error`` here, not as FP. Post-hoc
PH_CORR / PH_OTHER use the same 3-way validity label as an in-window
interruption.

Correct audio (paper Table 8): the first mid-speech non-backchannel
utterance gets the 4-way label. (B) affirm, (C) false_claim and
(D) unrelated are FP. On (A), the post-user-end response gets the 3-way
validity label: Invalid is FP ``ph_claim``; anything else, or silence, is TN.

Items whose judge labels are missing get ``result_type = None`` and a
``needs_*`` sub-type (or ``parse_failed``). They are never counted as
TP/FN/TN/FP.
"""
from __future__ import annotations

import re

EXTRA_WINDOW_S = 10.0

# Paper names of the turn-based ablations, keyed by the CLI value.
TURN_BASED_SETTINGS = {"truncated": "Truncated ablation", "full": "Full ablation"}


def _verdict(judgement: str | None, choices: str) -> str | None:
    """The label in parentheses, e.g. "(2)" or "(B)", that the judge concluded with.

    Reads the last "Final verdict" line in any of the forms the judge uses
    ("Final verdict: (2)", "Final verdict: **Invalid (2)**"). Without such a
    line, takes the last label in the text, since earlier ones are usually
    the analysis discussing the options.
    """
    if not judgement:
        return None
    label = rf"\(\s*([{choices}])\s*\)"
    for line in reversed(re.findall(r"Final verdict[^\n]*", judgement, re.I)):
        found = re.findall(label, line, re.I)
        if found:
            return found[-1].upper()
    found = re.findall(label, judgement, re.I)
    return found[-1].upper() if found else None


def parse_streaming_validity(judgement: str | None) -> str | None:
    """Map a streaming validity judgement to valid / invalid / unrelated."""
    label = _verdict(judgement, "123")
    return {"1": "valid", "2": "invalid", "3": "unrelated"}.get(label)


def parse_turn_validity(judgement: str | None) -> str | None:
    """Map a turn-based last-response judgement to valid / invalid / unrelated."""
    label = _verdict(judgement, "ABC")
    return {"A": "valid", "B": "invalid", "C": "unrelated"}.get(label)


def parse_four_way(judgement: str | None) -> str | None:
    """Map a 4-way judgement to A / B / C / D."""
    return _verdict(judgement, "ABCD")


def parse_backchannel(judgement: str | None) -> str | None:
    """Map a backchannel judgement to ``backchannel`` / ``non_backchannel``."""
    if not judgement:
        return None
    labels = re.findall(r"\(\s*([12])\s*\)", judgement)
    if not labels:
        return None
    return "backchannel" if labels[-1] == "1" else "non_backchannel"


def parse_timestamp(value) -> float | None:
    """Parse an error-step start time. The mapping prompt answers -1 when it
    cannot find the step; that, and anything unparsable, is None."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        t = float(value)
    else:
        match = re.search(r"-?\d+(?:\.\d+)?", str(value))
        if not match:
            return None
        t = float(match.group(0))
    return t if t >= 0 else None


def _unscored_fn(reason: str) -> dict:
    return {"result_type": None, "fn_sub_type": reason, "fp_sub_type": None}


def assign_streaming_error(
    *,
    has_mid_speech_non_bc: bool | None,
    first_non_bc_time: float | None,
    error_start: float | None,
    error_end: float | None,
    in_window_validity: str | None,
    posthoc_text: str | None,
    posthoc_validity: str | None,
) -> dict:
    """Assign TP or an FN sub-type for one incorrect-audio item.

    ``has_mid_speech_non_bc`` comes from the backchannel judge; None means it
    has not run. ``in_window_validity`` and ``posthoc_validity`` are
    ``valid``, ``invalid``, ``unrelated``, or None. Timing failures are not
    split by content. A label that cannot be assigned without a missing
    judge result is left unscored (``needs_*``), never guessed.
    """
    if has_mid_speech_non_bc is None:
        return _unscored_fn("needs_backchannel_judge")
    if has_mid_speech_non_bc:
        if first_non_bc_time is None:
            return _unscored_fn("needs_backchannel_judge")
        if error_start is None:
            return _unscored_fn("needs_error_timestamp")
        end = error_start if error_end is None else error_end
        window_end = end + EXTRA_WINDOW_S
        t = float(first_non_bc_time)
        if t < float(error_start):
            return {"result_type": "FN", "fn_sub_type": "too_early", "fp_sub_type": None}
        if t > window_end:
            return {"result_type": "FN", "fn_sub_type": "too_late", "fp_sub_type": None}
        if in_window_validity is None:
            return _unscored_fn("needs_validity_judge")
        if in_window_validity == "valid":
            return {"result_type": "TP", "fn_sub_type": None, "fp_sub_type": None}
        return {"result_type": "FN", "fn_sub_type": "wrong_error", "fp_sub_type": None}

    text = (posthoc_text or "").strip()
    if not text:
        return {"result_type": "FN", "fn_sub_type": "missed", "fp_sub_type": None}
    if posthoc_validity == "valid":
        return {"result_type": "FN", "fn_sub_type": "ph_corr", "fp_sub_type": None}
    if posthoc_validity in ("invalid", "unrelated"):
        return {"result_type": "FN", "fn_sub_type": "ph_other", "fp_sub_type": None}
    return _unscored_fn("needs_posthoc_judge")


def interruption_latency(t: float | None, error_start: float | None,
                         error_end: float | None, user_end: float | None) -> float | None:
    """Paper latency: max(0, t - t_e) if t_s <= t <= t_u, otherwise undefined.

    ``t`` is the first mid-speech non-backchannel utterance on incorrect
    audio. An interruption during the error step has latency 0; one after
    it is penalized by how far past t_e it lands. Too-early interruptions
    and anything without the needed times are undefined (None).
    """
    if t is None or error_start is None or user_end is None:
        return None
    end = error_start if error_end is None else error_end
    if not error_start <= t <= user_end:
        return None
    return max(0.0, t - end)


FOUR_WAY_FP = {"B": "affirm", "C": "false_claim", "D": "unrelated"}


def assign_streaming_correct(
    *,
    has_mid_speech_non_bc: bool | None,
    four_way: str | None,
    posthoc_text: str | None,
    posthoc_validity: str | None,
) -> dict:
    """Assign TN or an FP sub-type for one correct-audio item (paper Table 8).

    ``has_mid_speech_non_bc`` comes from the backchannel judge. When it is
    False the 4-way label is (A) without a judge call: the model stayed
    silent or only backchanneled. ``four_way`` is A / B / C / D or None.
    ``posthoc_validity`` is ``valid``, ``invalid``, ``unrelated``, or None.
    """
    if has_mid_speech_non_bc is None:
        return {"result_type": None, "fn_sub_type": None, "fp_sub_type": "needs_backchannel_judge"}
    label = "A" if not has_mid_speech_non_bc else four_way
    if label is None:
        return {"result_type": None, "fn_sub_type": None, "fp_sub_type": "needs_four_way_judge"}
    if label in FOUR_WAY_FP:
        return {"result_type": "FP", "fn_sub_type": None, "fp_sub_type": FOUR_WAY_FP[label]}

    text = (posthoc_text or "").strip()
    if not text:
        return {"result_type": "TN", "fn_sub_type": None, "fp_sub_type": None}
    if posthoc_validity is None:
        return {"result_type": None, "fn_sub_type": None, "fp_sub_type": "needs_posthoc_judge"}
    if posthoc_validity == "invalid":
        return {"result_type": "FP", "fn_sub_type": None, "fp_sub_type": "ph_claim"}
    return {"result_type": "TN", "fn_sub_type": None, "fp_sub_type": None}
