import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from mib_eval import judge as judge_module


class FakeClient:
    """Stands in for openai.OpenAI; returns the queued answers in order."""

    answers: list = []
    calls: list = []

    def __init__(self, api_key=None):
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))

    def create(self, **params):
        FakeClient.calls.append(params)
        text = FakeClient.answers.pop(0)
        choice = types.SimpleNamespace(message=types.SimpleNamespace(content=text), finish_reason="stop")
        return types.SimpleNamespace(choices=[choice])


class JudgeTest(unittest.TestCase):
    def setUp(self):
        FakeClient.answers, FakeClient.calls = [], []
        patches = [
            mock.patch.dict(sys.modules, {"openai": types.SimpleNamespace(OpenAI=FakeClient)}),
            mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test"}),
            mock.patch.object(judge_module.time, "sleep", lambda s: None),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_empty_answer_is_retried_and_not_cached(self):
        FakeClient.answers = ["", "Final verdict: (1)"]
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache.jsonl"
            judge = judge_module.make_judge(cache, model="gpt-5.2-2025-12-11")
            self.assertEqual(judge("prompt"), "Final verdict: (1)")
            rows = [json.loads(line) for line in cache.read_text().splitlines()]
        self.assertEqual(len(FakeClient.calls), 2)
        self.assertEqual([r["response"] for r in rows], ["Final verdict: (1)"])
        # Reasoning models get no temperature.
        self.assertNotIn("temperature", FakeClient.calls[0])

    def test_only_empty_answers_raise(self):
        FakeClient.answers = [""] * 5
        judge = judge_module.make_judge(None)
        with self.assertRaises(RuntimeError):
            judge("prompt")

    def test_cache_survives_a_truncated_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache.jsonl"
            FakeClient.answers = ["(2)"]
            judge_module.make_judge(cache)("prompt")
            with cache.open("a") as f:
                f.write('{"key": "abc", "resp')  # interrupted write
            judge = judge_module.make_judge(cache)
            self.assertEqual(judge("prompt"), "(2)")  # served from cache
        self.assertEqual(len(FakeClient.calls), 1)


if __name__ == "__main__":
    unittest.main()
