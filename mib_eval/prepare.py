"""Build the full-duplex result JSON from ASR transcripts and annotations.

This is the step between ``python -m mib_eval.asr`` and ``score-full-duplex``.
For every item directory ``<asr_root>/<id>/`` (``input.json`` = padded user
audio, ``output.json`` = model audio) it:

1. finds where the user's solution starts and ends in the user transcript,
2. merges the model's words into utterances and splits them into mid-speech
   utterances and post-user-end text,
3. maps the start of the error step to a time with the timestamp prompt
   (t_s) and aligns its end on the transcript (t_e),
4. runs the backchannel judge on the mid-speech utterances, in order, until
   the first non-backchannel one.

The in-window validity, 4-way and post-hoc labels are judged later by
``score-full-duplex``. An item that cannot be read is kept with a
``parse_error`` so it is reported as unscored instead of disappearing from
the denominators.
"""
from __future__ import annotations

import ast
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .taxonomy import parse_backchannel, parse_timestamp
from .window import align_error_step_end, normalize_word

# A pause of at least this long between model words starts a new utterance.
UTTERANCE_GAP_S = 1.0
SOLUTION_START_BUFFER_S = 1.0

# Spoken cue that precedes the user's solution, as normalized words.
CUE = ("here", "is", "my", "answer")


# ---------------------------------------------------------------- annotations

def load_annotations(path: Path) -> dict[int, dict]:
    return {int(row["index"]): row for row in json.loads(Path(path).read_text())}


def _spoken_steps(rewritten: str) -> dict[int, str]:
    """Split the colloquial rewrite on its ["Step N"] markers."""
    parts = re.split(r'\[\s*"Step\s*(\d+)"\s*\]', rewritten or "")
    return {int(n): text.strip() for n, text in zip(parts[1::2], parts[2::2])}


def annotation_fields(annotations: dict[int, dict], item_id: int) -> dict:
    """Ground-truth fields for one item.

    ``ground_truth_error_step_text`` is what the validity judge sees.
    ``error_step_alignment_text`` is the spoken error step alone, used for
    the timestamp prompt and the t_e alignment.
    """
    ann = annotations[item_id]
    has_error = ann["entry_type"] == "incorrect"
    error_text = alignment = None
    if has_error:
        steps = ast.literal_eval(ann["model_output_steps"])
        k = int(ann["model_output_solution_first_error_step"])
        # The judge also sees the following step, as in the original runs.
        error_text = " ".join(steps[k - 1:k + 1])
        alignment = _spoken_steps(ann["rewritten_solution"]).get(k)
        if not alignment:
            alignment = re.sub(r"^\s*Step\s*\d+\s*:\s*", "", steps[k - 1])
    return {
        "has_error": has_error,
        "question": ann["question"],
        "ground_truth_solution": ann["ground_truth_solution"],
        "complete_user_text_intended_to_say": re.sub(r"\[.*?\]", "", ann["rewritten_solution"]).strip(),
        "ground_truth_error_step_text": error_text,
        "error_step_alignment_text": alignment,
        "model_output_solution_first_error_reason": ann["model_output_solution_first_error_reason"] if has_error else None,
    }


# -------------------------------------------------------------- transcripts

def find_solution_start(chunks: list[dict]) -> int | None:
    """Index of the first word after the spoken cue "Here is my answer".

    Exact match first (case- and punctuation-insensitive, "Here's" allowed),
    then a match with one wrong word as long as the last cue word is right,
    which recovers ASR slips such as "Here is why answer".
    """
    words = [normalize_word(c.get("text") or "") for c in chunks]
    forms = [CUE, ("heres",) + CUE[2:]]

    for allowed_misses in (0, 1):
        for form in forms:
            n = len(form)
            for i in range(len(words) - n + 1):
                window = words[i:i + n]
                if window[0] != form[0] or window[-1] != form[-1]:
                    continue
                if sum(a != b for a, b in zip(window, form)) <= allowed_misses:
                    return i + n
    return None


def merge_utterances(chunks: list[dict], gap: float = UTTERANCE_GAP_S) -> list[dict]:
    """Merge ASR words into utterances, splitting at pauses of ``gap`` seconds."""
    utterances: list[dict] = []
    for chunk in chunks:
        start, end = chunk["timestamp"]
        if utterances and start - utterances[-1]["end"] < gap:
            utterances[-1]["text"] += " " + chunk["text"]
            utterances[-1]["end"] = end
        else:
            utterances.append({"text": chunk["text"], "start": start, "end": end})
    for u in utterances:
        u["text"] = u["text"].strip()
    return utterances


def parse_item(item_dir: Path, annotations: dict[int, dict]) -> dict:
    """Everything that needs no judge. Sets ``parse_error`` on failure."""
    item_id = int(item_dir.name)
    record: dict = {"id": item_id}
    try:
        record.update(annotation_fields(annotations, item_id))
    except (KeyError, IndexError, ValueError, SyntaxError) as exc:
        record["has_error"] = None
        record["parse_error"] = f"annotation: {exc!r}"
        return record

    try:
        user = json.loads((item_dir / "input.json").read_text())
        model = json.loads((item_dir / "output.json").read_text())
    except (OSError, ValueError) as exc:
        record["parse_error"] = f"transcript: {exc}"
        return record
    user_chunks = user.get("chunks") or []
    if not user_chunks:
        record["parse_error"] = "transcript: input.json has no words"
        return record

    start_idx = find_solution_start(user_chunks)
    if start_idx is None or start_idx >= len(user_chunks):
        record["parse_error"] = f"solution cue {' '.join(CUE)!r} not found in input.json"
        return record

    solution_start = user_chunks[start_idx]["timestamp"][0]
    solution_end = user_chunks[-1]["timestamp"][1]
    utterances = merge_utterances(model.get("chunks") or [])
    mid_speech = [u for u in utterances if solution_start + SOLUTION_START_BUFFER_S < u["start"] < solution_end]
    post_hoc = [u for u in utterances if u["start"] > solution_end]

    record.update({
        "solution_start_time": solution_start,
        "user_solution_end_time": solution_end,
        "mid_speech_utterances": mid_speech,
        "post_hoc_utterances": post_hoc,
        "model_output_after_solution_ends_text": " ".join(u["text"] for u in post_hoc),
        "_user_solution_chunks": user_chunks[start_idx:],
        "_input_data": user,
    })
    return record


def user_text_before(record: dict, t: float | None) -> str:
    chunks = record["_user_solution_chunks"]
    return " ".join(c["text"] for c in chunks if t is None or c["timestamp"][0] < t).strip()


# -------------------------------------------------------------------- judges

def judge_item(record: dict, *, judge, timestamp_template: str, backchannel_template: str) -> dict:
    """Timestamp mapping and backchannel labels for one parsed item."""
    if record.get("parse_error"):
        return record
    try:
        if record["has_error"] and record.get("error_step_alignment_text"):
            raw = judge(timestamp_template.format(
                input_data=json.dumps(record["_input_data"], ensure_ascii=False),
                ground_truth_error_step_text=record["error_step_alignment_text"],
            ))
            record["timestamp_mapping_judgement"] = raw
            record["ground_truth_error_step_timestamp"] = parse_timestamp(raw)
            record["ground_truth_error_step_end_timestamp"] = align_error_step_end(
                record["_input_data"].get("chunks") or [],
                record["ground_truth_error_step_timestamp"],
                record["error_step_alignment_text"],
            )

        first = None
        has_non_bc: bool | None = False
        for u in record["mid_speech_utterances"]:
            raw = judge(backchannel_template.format(
                question=record.get("question") or "",
                ground_truth_solution=record.get("ground_truth_solution") or "",
                user_solution_before_model_speaks=user_text_before(record, u["start"]),
                model_interruption_text=u["text"],
            ))
            u["backchannel_judgement"] = raw
            u["backchannel_label"] = parse_backchannel(raw)
            if u["backchannel_label"] is None:
                has_non_bc = None  # unreadable label: leave the item unscored
                break
            if u["backchannel_label"] == "non_backchannel":
                first, has_non_bc = u, True
                break

        # Paper MISSED / TN: a model that only backchannels after the user
        # finishes produced no reply. Judge the post-hoc utterances the same
        # way; if every one is a backchannel, the post-hoc text is empty.
        # Only needed when nothing non-backchannel happened mid-speech.
        if has_non_bc is False and record["post_hoc_utterances"]:
            all_backchannel = True
            for u in record["post_hoc_utterances"]:
                raw = judge(backchannel_template.format(
                    question=record.get("question") or "",
                    ground_truth_solution=record.get("ground_truth_solution") or "",
                    user_solution_before_model_speaks=user_text_before(record, None),
                    model_interruption_text=u["text"],
                ))
                u["backchannel_judgement"] = raw
                u["backchannel_label"] = parse_backchannel(raw)
                if u["backchannel_label"] != "backchannel":
                    # A reply, or an unreadable label: keep the text for the
                    # post-hoc validity judge rather than calling it silence.
                    all_backchannel = False
                    break
            if all_backchannel:
                record["model_output_after_solution_ends_raw_text"] = record["model_output_after_solution_ends_text"]
                record["model_output_after_solution_ends_text"] = ""
    except Exception as exc:  # judge failed after its retries
        record["judge_error"] = repr(exc)
        record["has_any_non_bc"] = None
        return record

    record["has_any_non_bc"] = has_non_bc
    record["first_non_bc_interruption_time"] = first["start"] if first else None
    record["first_non_bc_interruption_text"] = first["text"] if first else None
    record["user_solution_before_model_speaks"] = user_text_before(record, first["start"] if first else None)
    return record


def iter_item_dirs(asr_root: Path) -> list[Path]:
    dirs = [p for p in Path(asr_root).iterdir() if p.is_dir() and p.name.isdigit()]
    return sorted(dirs, key=lambda p: int(p.name))


def prepare_items(asr_root: Path, annotations_path: Path, *, judge,
                  timestamp_template: str, backchannel_template: str, workers: int = 8) -> list[dict]:
    annotations = load_annotations(annotations_path)
    records = [parse_item(d, annotations) for d in iter_item_dirs(asr_root)]

    def run(record):
        return judge_item(record, judge=judge, timestamp_template=timestamp_template,
                          backchannel_template=backchannel_template)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        records = list(pool.map(run, records))
    for record in records:
        for key in ("_user_solution_chunks", "_input_data"):
            record.pop(key, None)
    return records


def prepare_file(asr_root: Path, annotations_path: Path, output_path: Path, *,
                 judge, timestamp_path: Path, backchannel_path: Path, workers: int = 8) -> dict:
    if output_path.exists():
        raise ValueError(f"{output_path} already exists. Pass a new --output.")
    records = prepare_items(
        asr_root, annotations_path, judge=judge,
        timestamp_template=timestamp_path.read_text(),
        backchannel_template=backchannel_path.read_text(),
        workers=workers,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, indent=2, ensure_ascii=False))

    parse_errors = [r["id"] for r in records if r.get("parse_error")]
    judge_errors = [r["id"] for r in records if r.get("judge_error")]
    no_timestamp = [r["id"] for r in records
                    if r.get("has_error") and not r.get("parse_error")
                    and r.get("ground_truth_error_step_timestamp") is None]
    return {
        "n": len(records),
        "n_error": sum(1 for r in records if r.get("has_error")),
        "parse_errors": parse_errors,
        "judge_errors": judge_errors,
        "error_items_without_timestamp": no_timestamp,
    }
