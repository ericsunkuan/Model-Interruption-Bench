"""ASR alignment of the error-step end. Window is [start, end + 10s]."""
from __future__ import annotations

import re


def normalize_word(word: str) -> str:
    return re.sub(r"[^a-z0-9]", "", word.lower())


def align_error_step_end(
    chunks: list[dict],
    error_step_start_ts: float | None,
    error_step_text: str | None,
) -> float | None:
    """End time of the spoken error span on the padded user timeline.

    Starts at the ASR word closest to ``error_step_start_ts`` and walks
    forward matching the words of ``error_step_text`` (exact, one skipped
    word, or half the characters). If under 30% of the words match, the end
    is estimated at 0.4 s per word.
    """
    if error_step_start_ts is None or not chunks:
        return None

    start_ts = float(error_step_start_ts)
    target_words = [
        normalize_word(w)
        for w in (error_step_text or "").split()
        if normalize_word(w)
    ]
    if not target_words:
        return None

    start_idx = min(
        range(len(chunks)),
        key=lambda i: abs(chunks[i]["timestamp"][0] - start_ts),
    )
    target_ptr = 0
    last_matched_idx = start_idx
    limit = min(start_idx + len(target_words) * 3, len(chunks))
    for i in range(start_idx, limit):
        if target_ptr >= len(target_words):
            break
        chunk_word = normalize_word(chunks[i].get("text") or "")
        if not chunk_word:
            continue
        if chunk_word == target_words[target_ptr]:
            last_matched_idx = i
            target_ptr += 1
        elif target_ptr > 0 and chunk_word == target_words[min(target_ptr + 1, len(target_words) - 1)]:
            target_ptr += 2
            last_matched_idx = i
        elif len(chunk_word) > 2 and len(target_words[target_ptr]) > 2:
            common = sum(a == b for a, b in zip(chunk_word, target_words[target_ptr]))
            if common >= max(len(chunk_word), len(target_words[target_ptr])) * 0.5:
                last_matched_idx = i
                target_ptr += 1

    if target_ptr < len(target_words) * 0.3:
        return start_ts + len(target_words) * 0.4
    return chunks[last_matched_idx]["timestamp"][1]
