"""OpenAI judge shared by all prompts (timestamp, backchannel, validity, 4-way).

Decoding mirrors the original evaluation: seed 42, a single completion,
temperature 0 for non-reasoning models. Reasoning models (gpt-5*, o1, o3)
do not accept a temperature, so none is sent for them.

Responses are cached in a JSONL file keyed by model and prompt, so a rerun
reuses earlier judgements. The judge is safe to call from several threads.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path

REASONING_MARKERS = ("gpt-5", "o1-", "o3-", "o4-")


def _is_reasoning(model: str) -> bool:
    return any(marker in model.lower() for marker in REASONING_MARKERS)


def make_judge(
    cache_path: Path | None = None,
    model: str = "gpt-5.2-2025-12-11",
    max_tokens: int = 16384,
    retries: int = 5,
):
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set")
    client = OpenAI(api_key=api_key)

    cache: dict[str, str] = {}
    if cache_path and cache_path.exists():
        for line in cache_path.read_text().splitlines():
            try:
                row = json.loads(line)
                cache[row["key"]] = row["response"]
            except (ValueError, KeyError, TypeError):
                continue  # blank, or cut short by an interrupted run
    lock = threading.Lock()

    def judge(prompt: str) -> str:
        key = hashlib.sha256(f"{model}\n{prompt}".encode()).hexdigest()
        with lock:
            if key in cache:
                return cache[key]

        params = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "seed": 42,
        }
        if _is_reasoning(model):
            params["max_completion_tokens"] = max_tokens
        else:
            params["max_tokens"] = max_tokens
            params["temperature"] = 0
            params["top_p"] = 1.0

        delay = 1.0
        for attempt in range(retries):
            try:
                response = client.chat.completions.create(**params)
                text = (response.choices[0].message.content or "").strip()
                if not text:
                    # e.g. a reasoning model that spent its token budget
                    # thinking. Retry instead of caching an empty answer.
                    raise ValueError(f"empty response (finish_reason={response.choices[0].finish_reason})")
                break
            except Exception as exc:  # network, rate limit, server errors, empty answers
                if attempt == retries - 1:
                    raise RuntimeError(f"Judge call failed after {retries} attempts: {exc}") from exc
                time.sleep(delay)
                delay = min(delay * 2, 30.0)

        with lock:
            cache[key] = text
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with cache_path.open("a") as f:
                    f.write(json.dumps({"key": key, "response": text}) + "\n")
        return text

    return judge
