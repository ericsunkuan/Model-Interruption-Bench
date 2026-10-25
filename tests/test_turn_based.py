import json
import tempfile
import unittest
from pathlib import Path

from mib_eval.turn_based import judge_turn_item, score_turn_based, summarize_turn_based
from test_prepare import GSM8K_ANN, words

PROMPT = Path(__file__).resolve().parents[1] / "prompts" / "validity_last_response.txt"
SOLUTION = ("Here is the question. How many? Here is my answer. Four times five is twenty. "
            "Twenty plus five is twenty five. The answer is twenty five.")  # user ends at 12.9 s


def build(root: Path, item_id: int, model_chunks, user_text=SOLUTION):
    d = root / str(item_id)
    d.mkdir()
    (d / "input.json").write_text(json.dumps({"text": user_text, "chunks": words(user_text, 0)}))
    (d / "output.json").write_text(json.dumps({"text": "", "chunks": model_chunks}))


class TurnBasedTest(unittest.TestCase):
    def run_setting(self, setting, items, judge):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ann = root / "ann.json"
            ann.write_text(json.dumps(GSM8K_ANN))
            asr = root / "asr"
            asr.mkdir()
            for item_id, chunks in items.items():
                build(asr, item_id, chunks)
            out = root / "out.json"
            summary = score_turn_based(asr, ann, out, setting=setting, judge=judge, template_path=PROMPT)
            return summary, json.loads(out.read_text())

    def test_full_ablation_judges_post_turn_response(self):
        calls = []

        def judge(prompt):
            calls.append(prompt)
            return "Final verdict: (A) Valid"

        summary, records = self.run_setting("full", {
            0: words("mm", 8.0) + words("you should add six not five", 15.0),
            1: [],  # correct item, model never answers
        }, judge)

        self.assertEqual(len(calls), 1)
        self.assertIn("you should add six not five", calls[0])
        self.assertNotIn("\nmm you", calls[0])  # speech during the user turn is not the response
        self.assertEqual(records[0]["model_speech_during_user_turn"], "mm")
        self.assertEqual(records[1]["turn_label"], "no_response")
        self.assertEqual(summary["metrics"], {"TP%": 100.0, "TN%": 0.0})

    def test_truncated_ablation_scores_incorrect_items_only(self):
        summary, records = self.run_setting("truncated", {
            0: words("that should be plus six", 15.0),
            1: words("looks right", 15.0),
        }, lambda p: "Final verdict: (B)")
        self.assertEqual([r["id"] for r in records], [0])
        self.assertEqual(summary["excluded_correct_items"], 1)
        self.assertEqual(summary["metrics"], {"TP%": 0.0})

    def test_unscored_blocks_metrics(self):
        record = judge_turn_item({"id": 0, "has_error": True, "parse_error": "cue"}, template="", judge=None)
        summary = summarize_turn_based([record], "full")
        self.assertEqual(summary["unscored"], {"parse_failed": 1})
        self.assertIsNone(summary["metrics"])


if __name__ == "__main__":
    unittest.main()
